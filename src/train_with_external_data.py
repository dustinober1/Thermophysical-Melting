"""
Training with external Bradley datasets and extensive molecular fingerprints.
Based on successful Kaggle notebooks achieving MAE < 10.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys, RDKFingerprint, rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from rdkit.Avalon import pyAvalonTools
    avalon_available = True
except ImportError:
    avalon_available = False
    print("Warning: Avalon fingerprints not available")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"

RANDOM_STATE = 42


def load_data_with_external(use_external: bool = True):
    """Load train/test data and optionally Bradley external datasets"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    if use_external:
        # Note: These would need to be downloaded separately
        # From: https://www.kaggle.com/datasets/your-dataset/melting-point-chemical-dataset
        try:
            bradley_path = DATA_DIR / "BradleyMeltingPointDataset.xlsx"
            bradleyplus_path = DATA_DIR / "BradleyDoublePlusGoodMeltingPointDataset.xlsx"
            
            if bradley_path.exists() and bradleyplus_path.exists():
                bradley_df = pd.read_excel(bradley_path)
                bradleyplus_df = pd.read_excel(bradleyplus_path)
                
                # Convert Celsius to Kelvin
                bradley_df['Tm'] = bradley_df['mpC'] + 273.15
                bradleyplus_df['Tm'] = bradleyplus_df['mpC'] + 273.15
                
                bradley_df = bradley_df[['smiles', 'Tm']].rename(columns={'smiles': 'SMILES'})
                bradleyplus_df = bradleyplus_df[['smiles', 'Tm']].rename(columns={'smiles': 'SMILES'})
                
                # Merge with train
                external_df = pd.concat([bradley_df, bradleyplus_df], axis=0)
                train = pd.concat([train[['SMILES', 'Tm']], external_df], axis=0)
                train = train.drop_duplicates(subset=['SMILES', 'Tm']).reset_index(drop=True)
                
                print(f"✓ Loaded external data. Total training samples: {len(train)}")
            else:
                print("⚠ External datasets not found. Using original train data only.")
        except Exception as e:
            print(f"⚠ Could not load external data: {e}")
    
    return train, test


def extract_all_descriptors(smiles_list):
    """Extract all RDKit descriptors (217 features)"""
    descriptor_list = Descriptors._descList
    results = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            row = {name: 0 for name, func in descriptor_list}
        else:
            row = {name: func(mol) for name, func in descriptor_list}
        results.append(row)
    
    return pd.DataFrame(results)


def extract_all_fingerprints(smiles_list, morgan_nbits=1024, fcfp_nbits=1024, atompair_nbits=2048):
    """Extract comprehensive molecular fingerprints (6400+ features)"""
    fps_data = []
    
    # Initialize generators
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=morgan_nbits, countSimulation=True)
    fcfp = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fcfp_nbits, 
                                                          atomInvariantsGenerator=fcfp, countSimulation=True)
    atom_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=atompair_nbits, countSimulation=True)
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        feature_row = {}
        
        if mol is None:
            # Fill with zeros if invalid
            for i in range(morgan_nbits):
                feature_row[f"Morgan_{i}"] = 0
            for i in range(fcfp_nbits):
                feature_row[f"FCFP_{i}"] = 0
            for i in range(167):
                feature_row[f"MACCS_{i}"] = 0
            for i in range(atompair_nbits):
                feature_row[f"AtomPair_{i}"] = 0
            for i in range(2048):
                feature_row[f"RDKIT_{i}"] = 0
            if avalon_available:
                for i in range(1024):
                    feature_row[f"Avalon_{i}"] = 0
        else:
            # Morgan fingerprint (ECFP)
            morgan_fp = morgan_gen.GetFingerprint(mol)
            for i in range(morgan_nbits):
                feature_row[f"Morgan_{i}"] = morgan_fp[i]
            
            # Functional-class fingerprint (FCFP)
            fc_fp = fcfp_gen.GetFingerprint(mol)
            for i in range(fcfp_nbits):
                feature_row[f"FCFP_{i}"] = fc_fp[i]
            
            # MACCS keys (167 bits)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            for i in range(len(maccs_fp)):
                feature_row[f"MACCS_{i}"] = int(maccs_fp[i])
            
            # AtomPair fingerprint
            atompair_fp = atom_gen.GetCountFingerprint(mol)
            for i in range(atompair_nbits):
                feature_row[f"AtomPair_{i}"] = atompair_fp[i]
            
            # RDKIT fingerprint
            rdkit_fp = RDKFingerprint(mol, fpSize=2048)
            for i in range(len(rdkit_fp)):
                feature_row[f"RDKIT_{i}"] = int(rdkit_fp[i])
            
            # Avalon fingerprint (if available)
            if avalon_available:
                avalon_fp = pyAvalonTools.GetAvalonFP(mol, 1024)
                for i in range(len(avalon_fp)):
                    feature_row[f"Avalon_{i}"] = int(avalon_fp[i])
        
        fps_data.append(feature_row)
    
    return pd.DataFrame(fps_data)


def build_comprehensive_features(train_df, test_df):
    """Build all features: descriptors + fingerprints"""
    print("Extracting RDKit descriptors...")
    train_desc = extract_all_descriptors(train_df['SMILES'].tolist())
    test_desc = extract_all_descriptors(test_df['SMILES'].tolist())
    
    print("Extracting molecular fingerprints...")
    train_fps = extract_all_fingerprints(train_df['SMILES'].tolist())
    test_fps = extract_all_fingerprints(test_df['SMILES'].tolist())
    
    # Combine all features
    X_train = pd.concat([train_desc, train_fps], axis=1)
    X_test = pd.concat([test_desc, test_fps], axis=1)
    
    # Handle any NaN/inf values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Total features: {X_train.shape[1]}")
    return X_train, X_test


def train_with_feature_selection(
    X, y, X_test,
    model_type='lightgbm',
    n_folds=10,
    seed=RANDOM_STATE,
    use_yeo_johnson=True,
    use_feature_selection=True,
):
    """Train with Yeo-Johnson transformation and feature selection"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    test_preds = []
    
    yeo = PowerTransformer(method='yeo-johnson') if use_yeo_johnson else None
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        
        X_tr, X_val = X.iloc[trn_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[trn_idx].copy(), y.iloc[val_idx].copy()
        
        # Apply Yeo-Johnson transformation
        if yeo:
            y_tr = yeo.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
            y_val_transformed = yeo.transform(y_val.values.reshape(-1, 1)).ravel()
        else:
            y_val_transformed = y_val.values
        
        # Feature selection using a small model
        if use_feature_selection:
            print("Selecting important features...")
            selector_model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1
            )
            selector_model.fit(X_tr, y_tr)
            selector = SelectFromModel(selector_model, prefit=True, threshold="mean")
            
            selected_idx = selector.get_support(indices=True)
            selected_features = X_tr.columns[selected_idx]
            print(f"Selected {len(selected_features)} important features")
            
            X_tr_selected = X_tr[selected_features]
            X_val_selected = X_val[selected_features]
            X_test_selected = X_test[selected_features]
        else:
            X_tr_selected = X_tr
            X_val_selected = X_val
            X_test_selected = X_test
        
        # Train final model
        if model_type == 'lightgbm':
            params = {
                'objective': 'huber',
                'metric': 'mae',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'verbosity': -1,
                'random_state': seed,
                'n_jobs': -1,
            }
            
            train_data = lgb.Dataset(X_tr_selected, label=y_tr)
            val_data = lgb.Dataset(X_val_selected, label=y_val_transformed)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=15000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
            )
            
            val_pred = model.predict(X_val_selected, num_iteration=model.best_iteration)
            test_pred = model.predict(X_test_selected, num_iteration=model.best_iteration)
        
        elif model_type == 'xgboost':
            params = {
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'learning_rate': 0.05,
                'max_depth': 6,
                'tree_method': 'hist',
                'seed': seed,
                'verbosity': 0,
            }
            
            dtrain = xgb.DMatrix(X_tr_selected, label=y_tr)
            dval = xgb.DMatrix(X_val_selected, label=y_val_transformed)
            dtest = xgb.DMatrix(X_test_selected)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=15000,
                evals=[(dval, 'valid')],
                early_stopping_rounds=200,
                verbose_eval=500,
            )
            
            val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        
        else:  # catboost
            model = CatBoostRegressor(
                loss_function='MAE',
                learning_rate=0.05,
                depth=6,
                iterations=15000,
                random_seed=seed,
                verbose=500,
                task_type='CPU',
                od_type='Iter',
                od_wait=200,
            )
            
            train_pool = Pool(X_tr_selected, y_tr)
            val_pool = Pool(X_val_selected, y_val_transformed)
            
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            
            val_pred = model.predict(val_pool)
            test_pred = model.predict(X_test_selected)
        
        # Inverse transform
        if yeo:
            val_pred = yeo.inverse_transform(val_pred.reshape(-1, 1)).ravel()
            test_pred = yeo.inverse_transform(test_pred.reshape(-1, 1)).ravel()
        
        oof_preds[val_idx] = val_pred
        test_preds.append(test_pred)
        
        mae = mean_absolute_error(y_val, val_pred)
        print(f"Fold {fold} MAE: {mae:.4f}")
    
    # Average test predictions
    final_test_preds = np.mean(test_preds, axis=0)
    cv_mae = mean_absolute_error(y, oof_preds)
    
    print(f"\n{'='*60}")
    print(f"Overall CV MAE: {cv_mae:.4f}")
    print(f"{'='*60}")
    
    return oof_preds, final_test_preds, cv_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "catboost", "xgboost"], default="lightgbm")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--name", type=str, default="external_data_comprehensive")
    parser.add_argument("--use-external", action="store_true", help="Use Bradley external datasets")
    parser.add_argument("--no-yeo-johnson", action="store_true", help="Disable Yeo-Johnson transformation")
    parser.add_argument("--no-feature-selection", action="store_true", help="Disable feature selection")
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train, test = load_data_with_external(use_external=args.use_external)
    
    # Build comprehensive features
    print("\nBuilding comprehensive features...")
    X_train, X_test = build_comprehensive_features(train, test)
    y_train = train['Tm']
    
    # Drop any rows with NaN in target
    valid_idx = ~y_train.isna()
    X_train = X_train[valid_idx].reset_index(drop=True)
    y_train = y_train[valid_idx].reset_index(drop=True)
    
    print(f"\nFinal dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Train model
    oof, preds, cv_mae = train_with_feature_selection(
        X_train, y_train, X_test,
        model_type=args.model,
        n_folds=args.folds,
        seed=args.seed,
        use_yeo_johnson=not args.no_yeo_johnson,
        use_feature_selection=not args.no_feature_selection,
    )
    
    # Save submission
    SUB_DIR.mkdir(exist_ok=True)
    sub = pd.DataFrame({
        'id': test['id'],
        'Tm': preds
    })
    sub_path = SUB_DIR / f"submission_{args.name}_{args.model}_cv{cv_mae:.3f}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n✓ Saved submission to {sub_path}")


if __name__ == "__main__":
    main()
