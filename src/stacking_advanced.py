"""
Advanced stacking with multiple base models and diverse meta-learners.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import xgboost as xgb

from features import attach_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"
OOF_DIR = Path(__file__).resolve().parent.parent / "oof"

RANDOM_STATE = 42
N_FOLDS = 5


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def train_base_model(
    model_name: str,
    X_tr, y_tr, X_val, y_val, X_test,
    seed: int = 42,
    custom_params: Dict = None,
):
    """Train a single base model and return OOF and test predictions"""
    
    if model_name == "lightgbm":
        scaler = StandardScaler(with_mean=False)
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        lgb_train = lgb.Dataset(X_tr_scaled, label=y_tr)
        lgb_valid = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)
        
        if custom_params and "lightgbm" in custom_params:
            params = custom_params["lightgbm"]
        else:
            params = {
                "objective": "mae",
                "metric": "mae",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "min_data_in_leaf": 20,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "seed": seed,
                "num_threads": os.cpu_count() or 4,
            }
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        
        val_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)
        
    elif model_name == "xgboost":
        scaler = StandardScaler(with_mean=False)
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        dtrain = xgb.DMatrix(X_tr_scaled, label=y_tr)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        dtest = xgb.DMatrix(X_test_scaled)
        
        if custom_params and "xgboost" in custom_params:
            params = custom_params["xgboost"]
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
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dval, "valid")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        
    elif model_name == "catboost":
        train_pool = Pool(X_tr, y_tr)
        valid_pool = Pool(X_val, y_val)
        
        if custom_params and "catboost" in custom_params:
            model = CatBoostRegressor(**custom_params["catboost"], random_seed=seed, verbose=0)
        else:
            model = CatBoostRegressor(
                loss_function="MAE",
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3.0,
                iterations=20000,
                random_seed=seed,
                od_type="Iter",
                od_wait=500,
                task_type="CPU",
                verbose=0,
            )
        
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        val_pred = model.predict(valid_pool)
        test_pred = model.predict(X_test)
    
    elif model_name == "rf":
        # Random Forest as a diverse base learner
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    
    else:
        raise ValueError(f"Unknown base model: {model_name}")
    
    return val_pred, test_pred


def generate_base_predictions(
    base_models: List[str],
    X, y, X_test,
    n_folds: int = 5,
    seed: int = 42,
    custom_params: Dict = None,
):
    """Generate OOF and test predictions for all base models"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    n_train = len(X)
    n_test = len(X_test)
    n_models = len(base_models)
    
    # Initialize arrays for OOF and test predictions
    oof_preds = np.zeros((n_train, n_models))
    test_preds = np.zeros((n_test, n_models))
    
    for model_idx, model_name in enumerate(base_models):
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} ({model_idx + 1}/{n_models})")
        print(f"{'='*60}")
        
        fold_test_preds = np.zeros((n_test, n_folds))
        
        for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_tr, X_val = X[trn_idx], X[val_idx]
            y_tr, y_val = y[trn_idx], y[val_idx]
            
            val_pred, test_pred = train_base_model(
                model_name, X_tr, y_tr, X_val, y_val, X_test,
                seed=seed + fold,
                custom_params=custom_params,
            )
            
            oof_preds[val_idx, model_idx] = val_pred
            fold_test_preds[:, fold - 1] = test_pred
            
            mae = mean_absolute_error(y_val, val_pred)
            print(f"  Fold {fold}: MAE = {mae:.4f}")
        
        # Average test predictions across folds
        test_preds[:, model_idx] = fold_test_preds.mean(axis=1)
        
        # Overall OOF MAE for this model
        oof_mae = mean_absolute_error(y, oof_preds[:, model_idx])
        print(f"  OOF MAE: {oof_mae:.4f}")
    
    return oof_preds, test_preds


def train_meta_model(meta_learner: str, X_meta, y, alpha: float = 1.0):
    """Train meta-model on OOF predictions"""
    
    if meta_learner == "ridge":
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    elif meta_learner == "elasticnet":
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)
    elif meta_learner == "lasso":
        model = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=5000)
    elif meta_learner == "huber":
        model = HuberRegressor(epsilon=1.35, alpha=alpha, max_iter=500)
    elif meta_learner == "krr":
        # Kernel Ridge Regression - can capture non-linear relationships
        model = KernelRidge(alpha=alpha, kernel="rbf", gamma=0.1)
    elif meta_learner == "lightgbm":
        # Simple LightGBM as meta-learner
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            objective="mae",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown meta-learner: {meta_learner}")
    
    model.fit(X_meta, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--name", type=str, default="advanced_stack")
    parser.add_argument("--base-models", type=str, default="lightgbm,catboost,xgboost",
                       help="Comma-separated list of base models (lightgbm,catboost,xgboost,rf)")
    parser.add_argument("--meta-learner", type=str, default="ridge",
                       choices=["ridge", "elasticnet", "lasso", "huber", "krr", "lightgbm"])
    parser.add_argument("--meta-alpha", type=float, default=1.0,
                       help="Regularization parameter for meta-learner")
    
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
    
    # Custom params from JSON
    parser.add_argument("--params-json", type=str, help="Path to JSON file with model-specific params")
    
    args = parser.parse_args()
    
    # Parse base models
    base_models = [m.strip() for m in args.base_models.split(",")]
    print(f"Base models: {base_models}")
    print(f"Meta-learner: {args.meta_learner}")
    
    # Load custom params if provided
    custom_params = None
    if args.params_json:
        import json
        with open(args.params_json, "r") as f:
            custom_params = json.load(f)
        print(f"Loaded custom parameters from {args.params_json}")
    
    # Load data
    train, test = load_data()
    
    # Build features
    print("\nBuilding features...")
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
    
    X = X_df.values
    y = train["Tm"].values
    X_test = Xte_df.values
    
    # Generate base model predictions
    print("\n" + "="*60)
    print("PHASE 1: Generating base model predictions")
    print("="*60)
    
    oof_preds, test_preds = generate_base_predictions(
        base_models=base_models,
        X=X,
        y=y,
        X_test=X_test,
        n_folds=args.folds,
        seed=args.seed,
        custom_params=custom_params,
    )
    
    # Save OOF and test predictions
    OOF_DIR.mkdir(exist_ok=True)
    for i, model_name in enumerate(base_models):
        np.save(OOF_DIR / f"oof_{args.name}_{model_name}.npy", oof_preds[:, i])
        np.save(OOF_DIR / f"test_{args.name}_{model_name}.npy", test_preds[:, i])
    
    print(f"\nSaved OOF and test predictions to {OOF_DIR}")
    
    # Train meta-model
    print("\n" + "="*60)
    print("PHASE 2: Training meta-model")
    print("="*60)
    
    meta_model = train_meta_model(args.meta_learner, oof_preds, y, alpha=args.meta_alpha)
    meta_preds = meta_model.predict(test_preds)
    
    # Evaluate
    oof_meta = meta_model.predict(oof_preds)
    cv_mae = mean_absolute_error(y, oof_meta)
    
    print(f"\nMeta-model ({args.meta_learner}) CV MAE: {cv_mae:.4f}")
    
    # Print weights if available
    if hasattr(meta_model, 'coef_'):
        print("\nMeta-model weights:")
        for i, model_name in enumerate(base_models):
            print(f"  {model_name}: {meta_model.coef_[i]:.4f}")
        if hasattr(meta_model, 'intercept_'):
            print(f"  intercept: {meta_model.intercept_:.4f}")
    
    # Save submission
    SUB_DIR.mkdir(exist_ok=True)
    sub_path = SUB_DIR / f"submission_{args.name}_{args.meta_learner}_cv{cv_mae:.3f}.csv"
    sub = pd.DataFrame({"id": test["id"], "Tm": meta_preds})
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved submission to {sub_path}")
    
    # Also save simple average for comparison
    avg_preds = test_preds.mean(axis=1)
    oof_avg = oof_preds.mean(axis=1)
    avg_mae = mean_absolute_error(y, oof_avg)
    print(f"\nSimple average CV MAE: {avg_mae:.4f}")
    
    avg_sub_path = SUB_DIR / f"submission_{args.name}_avg_cv{avg_mae:.3f}.csv"
    avg_sub = pd.DataFrame({"id": test["id"], "Tm": avg_preds})
    avg_sub.to_csv(avg_sub_path, index=False)
    print(f"Saved average submission to {avg_sub_path}")


if __name__ == "__main__":
    main()
