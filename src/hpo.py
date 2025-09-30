"""
Hyperparameter optimization using Optuna for melting point prediction models.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from features import attach_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def objective_lgb(trial, X, y, n_folds=3, seed=42):
    """Optuna objective for LightGBM hyperparameter optimization"""
    params = {
        "objective": "mae",
        "metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "verbosity": -1,
        "seed": seed,
        "num_threads": os.cpu_count() or 4,
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores = []
    
    # Standardize features
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    
    for train_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        
        pred = model.predict(X_val, num_iteration=model.best_iteration)
        cv_scores.append(mean_absolute_error(y_val, pred))
    
    return np.mean(cv_scores)


def objective_xgb(trial, X, y, n_folds=3, seed=42):
    """Optuna objective for XGBoost hyperparameter optimization"""
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "seed": seed,
        "verbosity": 0,
        "n_jobs": os.cpu_count() or 4,
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores = []
    
    # Standardize features
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    
    for train_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        cv_scores.append(mean_absolute_error(y_val, pred))
    
    return np.mean(cv_scores)


def objective_catboost(trial, X, y, n_folds=3, seed=42):
    """Optuna objective for CatBoost hyperparameter optimization"""
    params = {
        "loss_function": "MAE",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "iterations": 2000,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 50,
        "task_type": "CPU",
        "verbose": False,
    }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        train_pool = cb.Pool(X_tr, y_tr)
        valid_pool = cb.Pool(X_val, y_val)
        
        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        
        pred = model.predict(X_val)
        cv_scores.append(mean_absolute_error(y_val, pred))
    
    return np.mean(cv_scores)


def optimize_model(model_type: str, X: np.ndarray, y: np.ndarray, n_trials: int = 100, n_folds: int = 3, seed: int = 42):
    """Run hyperparameter optimization for a specific model"""
    
    print(f"Optimizing {model_type} with {n_trials} trials...")
    
    if model_type == "lightgbm":
        objective_func = lambda trial: objective_lgb(trial, X, y, n_folds, seed)
    elif model_type == "xgboost":
        objective_func = lambda trial: objective_xgb(trial, X, y, n_folds, seed)
    elif model_type == "catboost":
        objective_func = lambda trial: objective_catboost(trial, X, y, n_folds, seed)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=n_trials)
    
    print(f"Best {model_type} score: {study.best_value:.4f}")
    print(f"Best {model_type} params: {study.best_params}")
    
    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for melting point prediction")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "catboost", "all"], default="lightgbm")
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds for optimization")
    parser.add_argument("--seed", type=int, default=42)
    
    # Feature flags (same as train.py)
    parser.add_argument("--smiles-basic", action="store_true")
    parser.add_argument("--smiles-tfidf", action="store_true")
    parser.add_argument("--chemical-structure", action="store_true")
    parser.add_argument("--advanced-features", action="store_true")
    parser.add_argument("--polynomial", action="store_true")
    parser.add_argument("--interactions", action="store_true")
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument("--tfidf-ngram-min", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=4)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    
    args = parser.parse_args()
    
    # Load data
    train, test = load_data()
    
    # Build features
    X_df, _, features = attach_features(
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
    )
    
    X = X_df.values
    y = train["Tm"].values
    
    print(f"Using {len(features)} features for optimization")
    
    if args.model == "all":
        models = ["lightgbm", "xgboost", "catboost"]
    else:
        models = [args.model]
    
    best_results = {}
    for model in models:
        best_params, best_score = optimize_model(model, X, y, args.trials, args.folds, args.seed)
        best_results[model] = {"params": best_params, "score": best_score}
    
    print("\n=== OPTIMIZATION RESULTS ===")
    for model, result in best_results.items():
        print(f"\n{model.upper()}:")
        print(f"  Best CV MAE: {result['score']:.4f}")
        print(f"  Best params: {result['params']}")


if __name__ == "__main__":
    main()