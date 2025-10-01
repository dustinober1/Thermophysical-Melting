"""
Advanced training script with target transformation and robust losses.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
from features import attach_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"

RANDOM_STATE = 42
N_FOLDS = 5


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def apply_target_transform(y_train, y_test=None, method="none"):
    """Apply target transformation"""
    if method == "none":
        return y_train, y_test, None
    elif method == "log1p":
        transformer = None
        y_train_transformed = np.log1p(y_train)
        y_test_transformed = np.log1p(y_test) if y_test is not None else None
        return y_train_transformed, y_test_transformed, "log1p"
    elif method == "yeo-johnson":
        transformer = PowerTransformer(method="yeo-johnson")
        y_train_transformed = transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_transformed = transformer.transform(y_test.reshape(-1, 1)).ravel() if y_test is not None else None
        return y_train_transformed, y_test_transformed, transformer
    else:
        raise ValueError(f"Unknown transform method: {method}")


def inverse_target_transform(y_pred, transformer):
    """Inverse transform predictions"""
    if transformer is None:
        return y_pred
    elif transformer == "log1p":
        return np.expm1(y_pred)
    else:  # PowerTransformer
        return transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()


def kfold_cv_advanced(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_type: str = "lightgbm",
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_STATE,
    target_transform: str = "none",
    use_sample_weights: bool = False,
    use_huber: bool = False,
    use_smiles_basic: bool = False,
    use_smiles_tfidf: bool = False,
    use_chemical_structure: bool = False,
    use_advanced_features: bool = False,
    use_polynomial: bool = False,
    use_interactions: bool = False,
    tfidf_ngram_min: int = 2,
    tfidf_ngram_max: int = 5,
    tfidf_min_df: int = 2,
    svd_components: int = 256,
    use_rdkit_desc: bool = False,
    use_morgan: bool = False,
    morgan_radius: int = 2,
    morgan_nbits: int = 1024,
    use_maccs: bool = False,
    # Custom hyperparameters
    custom_params: Optional[dict] = None,
):
    """Advanced K-fold CV with target transformation and robust options"""
    
    # Build features
    X_df, Xte_df, features = attach_features(
        train,
        test,
        use_smiles_basic=use_smiles_basic,
        use_smiles_tfidf=use_smiles_tfidf,
        use_chemical_structure=use_chemical_structure,
        use_advanced_features=use_advanced_features,
        use_polynomial=use_polynomial,
        use_interactions=use_interactions,
        tfidf_ngram_min=tfidf_ngram_min,
        tfidf_ngram_max=tfidf_ngram_max,
        tfidf_min_df=tfidf_min_df,
        svd_components=svd_components,
        random_state=seed,
        use_rdkit_desc=use_rdkit_desc,
        use_morgan=use_morgan,
        morgan_radius=morgan_radius,
        morgan_nbits=morgan_nbits,
        use_maccs=use_maccs,
    )
    
    X = X_df.values
    y = train["Tm"].values
    X_test = Xte_df.values
    
    # Apply target transformation
    y_transformed, _, transformer = apply_target_transform(y, method=target_transform)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    
    # Standardize for tree models if needed
    scaler = None
    if model_type in ["lightgbm", "xgboost"]:
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y_transformed), 1):
        X_tr, X_val = X[trn_idx], X[val_idx]
        y_tr, y_val = y_transformed[trn_idx], y_transformed[val_idx]
        
        # Sample weights (downweight outliers)
        sample_weight = None
        if use_sample_weights:
            # Compute residuals from median to identify outliers
            median_val = np.median(y_tr)
            residuals = np.abs(y_tr - median_val)
            threshold = np.percentile(residuals, 90)
            sample_weight = np.where(residuals > threshold, 0.5, 1.0)
        
        if model_type == "lightgbm":
            # Use custom params if provided, otherwise use defaults
            if custom_params:
                params = custom_params.copy()
            else:
                params = {
                    "objective": "huber" if use_huber else "mae",
                    "metric": "mae",
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "min_data_in_leaf": 20,
                    "lambda_l2": 1.0,
                    "verbosity": -1,
                    "seed": seed + fold,
                    "num_threads": os.cpu_count() or 4,
                }
                if use_huber:
                    params["alpha"] = 0.9  # Huber parameter
            
            lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
            lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
            )
            
            oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        
        elif model_type == "xgboost":
            if custom_params:
                params = custom_params.copy()
            else:
                params = {
                    "objective": "reg:absoluteerror",
                    "eval_metric": "mae",
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 10,
                    "reg_alpha": 1.0,
                    "reg_lambda": 1.0,
                    "seed": seed,
                    "verbosity": 0,
                    "n_jobs": os.cpu_count() or 4,
                }
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weight)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=5000,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=100,
                verbose_eval=100,
            )
            
            oof[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / n_folds
        
        elif model_type == "catboost":
            train_pool = Pool(X_tr, y_tr, weight=sample_weight)
            valid_pool = Pool(X_val, y_val)
            
            if custom_params:
                model = CatBoostRegressor(**custom_params, random_seed=seed + fold, verbose=200)
            else:
                model = CatBoostRegressor(
                    loss_function="MAE",
                    learning_rate=0.05,
                    depth=8,
                    l2_leaf_reg=3.0,
                    iterations=20000,
                    random_seed=seed + fold,
                    od_type="Iter",
                    od_wait=500,
                    task_type="CPU",
                    verbose=200,
                )
            
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            oof[val_idx] = model.predict(valid_pool)
            preds += model.predict(X_test) / n_folds
        
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Inverse transform for evaluation
        oof_original = inverse_target_transform(oof[val_idx], transformer)
        y_original = y[val_idx]
        
        mae = mean_absolute_error(y_original, oof_original)
        print(f"Fold {fold}: MAE = {mae:.4f}")
    
    # Inverse transform final predictions
    oof_final = inverse_target_transform(oof, transformer)
    preds_final = inverse_target_transform(preds, transformer)
    
    cv_mae = mean_absolute_error(y, oof_final)
    print(f"CV MAE: {cv_mae:.4f}")
    
    return oof_final, preds_final, cv_mae


def save_submission(test: pd.DataFrame, preds: np.ndarray, name: str):
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUB_DIR / f"submission_{name}.csv"
    sub = pd.DataFrame({"id": test["id"], "Tm": preds})
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "catboost", "xgboost"], default="lightgbm")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--name", type=str, default="advanced_experiment")
    
    # Advanced options
    parser.add_argument("--target-transform", choices=["none", "log1p", "yeo-johnson"], default="none")
    parser.add_argument("--sample-weights", action="store_true", help="Use sample weighting to downweight outliers")
    parser.add_argument("--huber", action="store_true", help="Use Huber loss (LightGBM only)")
    
    # Feature flags
    parser.add_argument("--smiles-basic", action="store_true")
    parser.add_argument("--smiles-tfidf", action="store_true")
    parser.add_argument("--chemical-structure", action="store_true")
    parser.add_argument("--advanced-features", action="store_true")
    parser.add_argument("--polynomial", action="store_true")
    parser.add_argument("--interactions", action="store_true")
    parser.add_argument("--tfidf-ngram-min", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=5)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--rdkit-desc", action="store_true")
    parser.add_argument("--morgan", action="store_true")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-nbits", type=int, default=1024)
    parser.add_argument("--maccs", action="store_true")
    
    # Custom hyperparameters from JSON file
    parser.add_argument("--params-json", type=str, help="Path to JSON file with custom hyperparameters")
    
    args = parser.parse_args()
    
    # Load custom params if provided
    custom_params = None
    if args.params_json:
        import json
        with open(args.params_json, "r") as f:
            custom_params = json.load(f)
        print(f"Loaded custom parameters from {args.params_json}")
    
    train, test = load_data()
    
    # Feature summary
    X_df, Xte_df, features = attach_features(
        train,
        test,
        use_smiles_basic=args.smiles_basic,
        use_smiles_tfidf=args.smiles_tfidf,
        use_chemical_structure=args.chemical_structure,
        use_advanced_features=args.advanced_features,
        use_polynomial=args.polynomial,
        use_interactions=args.interactions,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_min_df=args.tfidf_min_df,
        svd_components=args.svd_components,
        random_state=args.seed,
        use_rdkit_desc=args.rdkit_desc,
        use_morgan=args.morgan,
        morgan_radius=args.morgan_radius,
        morgan_nbits=args.morgan_nbits,
        use_maccs=args.maccs,
    )
    
    n_group = sum(c.startswith('Group ') for c in features)
    n_sbasic = sum(c.startswith('S_') for c in features)
    n_chem = sum(c.startswith('CHEM_') for c in features)
    n_adv = sum(c.startswith('ADV_') for c in features)
    n_tfidf = sum(c.startswith('TFIDF_SVD_') for c in features)
    n_rd = sum(c.startswith('RD_') for c in features)
    n_mg = sum(c.startswith('MG_') for c in features)
    n_maccs = sum(c.startswith('MACCS_') for c in features)
    
    print(
        f"Using {len(features)} features ("
        f"{n_group} Group, {n_sbasic} S-basic, {n_chem} Chem, {n_adv} Adv, "
        f"{n_tfidf} TFIDF, {n_rd} RDKit, {n_mg} Morgan, {n_maccs} MACCS)"
    )
    
    oof, preds, cv_mae = kfold_cv_advanced(
        train,
        test,
        model_type=args.model,
        n_folds=args.folds,
        seed=args.seed,
        target_transform=args.target_transform,
        use_sample_weights=args.sample_weights,
        use_huber=args.huber,
        use_smiles_basic=args.smiles_basic,
        use_smiles_tfidf=args.smiles_tfidf,
        use_chemical_structure=args.chemical_structure,
        use_advanced_features=args.advanced_features,
        use_polynomial=args.polynomial,
        use_interactions=args.interactions,
        tfidf_ngram_min=args.tfidf_ngram_min,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_min_df=args.tfidf_min_df,
        svd_components=args.svd_components,
        use_rdkit_desc=args.rdkit_desc,
        use_morgan=args.morgan,
        morgan_radius=args.morgan_radius,
        morgan_nbits=args.morgan_nbits,
        use_maccs=args.maccs,
        custom_params=custom_params,
    )
    
    save_submission(test, preds, f"{args.name}_{args.model}_cv{cv_mae:.3f}")


if __name__ == "__main__":
    main()
