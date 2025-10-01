"""
Hyperparameter optimization with Optuna for melting point prediction.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from features import attach_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42
N_FOLDS = 5


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def objective_lightgbm(trial: optuna.Trial, X, y, n_folds=5, seed=42):
    """Optuna objective for LightGBM"""
    params = {
        "objective": "mae",
        "metric": "mae",
        "verbosity": -1,
        "seed": seed,
        "num_threads": os.cpu_count() or 4,
        # Hyperparameters to tune
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    
    # Optional standardization
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    
    for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
        X_tr, X_val = X_scaled[trn_idx], X_scaled[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]
        
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    cv_mae = mean_absolute_error(y, oof)
    return cv_mae


def objective_catboost(trial: optuna.Trial, X, y, n_folds=5, seed=42):
    """Optuna objective for CatBoost"""
    params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": seed,
        "verbose": 0,
        "task_type": "CPU",
        "od_type": "Iter",
        "od_wait": 300,
        "iterations": 10000,
        # Hyperparameters to tune
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # random subspace method (feature fraction)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 4),
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    
    for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[trn_idx], X[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]
        
        train_pool = Pool(X_tr, y_tr)
        valid_pool = Pool(X_val, y_val)
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        
        oof[val_idx] = model.predict(valid_pool)
    
    cv_mae = mean_absolute_error(y, oof)
    return cv_mae


def objective_xgboost(trial: optuna.Trial, X, y, n_folds=5, seed=42):
    """Optuna objective for XGBoost"""
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "seed": seed,
        "verbosity": 0,
        "n_jobs": os.cpu_count() or 4,
        # Hyperparameters to tune
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    
    # Optional standardization
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    
    for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
        X_tr, X_val = X_scaled[trn_idx], X_scaled[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dval, "valid")],
            early_stopping_rounds=150,
            verbose_eval=0,
        )
        
        oof[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    
    cv_mae = mean_absolute_error(y, oof)
    return cv_mae


def run_optimization(
    model_type: str,
    X,
    y,
    n_trials: int = 100,
    n_folds: int = 5,
    seed: int = 42,
):
    """Run Optuna optimization for specified model"""
    print(f"\n{'='*60}")
    print(f"Running {n_trials} trials of HPO for {model_type.upper()}")
    print(f"{'='*60}\n")
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    if model_type == "lightgbm":
        objective_fn = lambda trial: objective_lightgbm(trial, X, y, n_folds, seed)
    elif model_type == "catboost":
        objective_fn = lambda trial: objective_catboost(trial, X, y, n_folds, seed)
    elif model_type == "xgboost":
        objective_fn = lambda trial: objective_xgboost(trial, X, y, n_folds, seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV MAE: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "catboost", "xgboost"], required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--name", type=str, default="hpo_experiment")
    
    # Feature flags (same as train.py)
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
    
    args = parser.parse_args()
    
    # Load data
    train, test = load_data()
    
    # Build features
    print("Building features...")
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
    print(
        f"Using {len(features)} features ("
        f"{n_group} Group, {n_sbasic} S-basic, {n_chem} Chem, {n_adv} Adv, {n_tfidf} TFIDF)"
    )
    
    X = X_df.values
    y = train["Tm"].values
    
    # Run optimization
    study = run_optimization(
        model_type=args.model,
        X=X,
        y=y,
        n_trials=args.trials,
        n_folds=args.folds,
        seed=args.seed,
    )
    
    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "optuna_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save study
    study_path = results_dir / f"{args.name}_{args.model}_study.pkl"
    import joblib
    joblib.dump(study, study_path)
    print(f"Saved study to {study_path}")
    
    # Save best params as JSON
    import json
    params_path = results_dir / f"{args.name}_{args.model}_best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved best params to {params_path}")
    
    # Save trials dataframe
    df_path = results_dir / f"{args.name}_{args.model}_trials.csv"
    study.trials_dataframe().to_csv(df_path, index=False)
    print(f"Saved trials to {df_path}")


if __name__ == "__main__":
    main()
