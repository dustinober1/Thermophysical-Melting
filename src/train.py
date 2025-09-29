import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from features import attach_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"

RANDOM_STATE = 42
N_FOLDS = 5


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # Deprecated: kept for compatibility if needed elsewhere
    group_cols = [c for c in df.columns if c.startswith("Group ")]
    return group_cols


def kfold_cv(train: pd.DataFrame, test: pd.DataFrame, model_type: str = "lightgbm", n_folds: int = N_FOLDS, seed: int = RANDOM_STATE, use_smiles_basic: bool = False):
    X_df, Xte_df, features = attach_features(train, test, use_smiles_basic=use_smiles_basic)
    X = X_df.values
    y = train["Tm"].values
    X_test = Xte_df.values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    # Standardize for LightGBM when using leaves-wise splits (optional, CatBoost doesn't need it)
    scaler = None
    if model_type == "lightgbm":
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X[trn_idx], X[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        if model_type == "lightgbm":
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
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
                "seed": seed + fold,
                "num_threads": os.cpu_count() or 4,
            }
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

        elif model_type == "catboost":
            train_pool = Pool(X_tr, y_tr)
            valid_pool = Pool(X_val, y_val)
            model = CatBoostRegressor(
                loss_function="MAE",
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3.0,
                iterations=20000,
                random_seed=seed + fold,
                od_type="Iter",
                od_wait=500,
                task_type="CPU",  # change to GPU if available
                verbose=200,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            oof[val_idx] = model.predict(valid_pool)
            preds += model.predict(X_test) / n_folds

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        mae = mean_absolute_error(y[val_idx], oof[val_idx])
        print(f"Fold {fold}: MAE = {mae:.4f}")

    cv_mae = mean_absolute_error(y, oof)
    print(f"CV MAE: {cv_mae:.4f}")
    return oof, preds, cv_mae


def save_submission(test: pd.DataFrame, preds: np.ndarray, name: str):
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUB_DIR / f"submission_{name}.csv"
    sub = pd.DataFrame({"id": test["id"], "Tm": preds})
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--name", type=str, default="baseline_group_features")
    parser.add_argument("--smiles-basic", action="store_true", help="Include simple SMILES text features")
    args = parser.parse_args()

    train, test = load_data()

    # Basic checks
    assert "Tm" in train.columns, "Target Tm missing in train.csv"
    # Build features
    X_df, Xte_df, features = attach_features(train, test, use_smiles_basic=args.smiles_basic)
    assert len(features) > 0, "No features found"
    print(f"Using {len(features)} features ({sum(c.startswith('Group ') for c in features)} Group + {sum(c.startswith('S_') for c in features)} SMILES-basic)")
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")

    oof, preds, cv_mae = kfold_cv(train, test, model_type=args.model, n_folds=args.folds, seed=args.seed, use_smiles_basic=args.smiles_basic)
    save_submission(test, preds, f"{args.name}_{args.model}_cv{cv_mae:.3f}")


if __name__ == "__main__":
    main()
