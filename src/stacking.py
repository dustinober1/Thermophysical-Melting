import argparse
from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

from features import attach_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"
OOF_DIR = Path(__file__).resolve().parent.parent / "oof"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def fit_lgbm(X, y, X_val, y_val, seed: int):
    lgb_train = lgb.Dataset(X, label=y)
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
        "seed": seed,
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
    return model


def fit_catboost(X, y, X_val, y_val, seed: int):
    train_pool = Pool(X, y)
    valid_pool = Pool(X_val, y_val)
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
        verbose=200,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def make_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    folds: int,
    seed: int,
    models: List[str],
):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_preds: Dict[str, np.ndarray] = {m: np.zeros(len(y)) for m in models}
    test_preds: Dict[str, np.ndarray] = {m: np.zeros(X_test.shape[0]) for m in models}

    # Standardize for LGBM (and Ridge later)
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X_scaled[trn_idx], X_scaled[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        if "lightgbm" in models:
            m = fit_lgbm(X_tr, y_tr, X_val, y_val, seed + fold)
            oof_preds["lightgbm"][val_idx] = m.predict(X_val, num_iteration=m.best_iteration)
            test_preds["lightgbm"] += m.predict(X_test_scaled, num_iteration=m.best_iteration) / folds

        if "catboost" in models:
            m = fit_catboost(X_tr, y_tr, X_val, y_val, seed + fold)
            oof_preds["catboost"][val_idx] = m.predict(X_val)
            test_preds["catboost"] += m.predict(X_test_scaled) / folds

        print(
            f"Fold {fold} | "
            + " ".join([f"{m}: {mean_absolute_error(y[val_idx], oof_preds[m][val_idx]):.4f}" for m in models])
        )

    cv_scores = {m: mean_absolute_error(y, oof_preds[m]) for m in models}
    print("Base CV MAEs:", cv_scores)

    return oof_preds, test_preds, cv_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="stacking")
    parser.add_argument("--models", type=str, default="lightgbm,catboost", help="Comma list of base models")
    # Feature flags (same as train.py)
    parser.add_argument("--smiles-basic", action="store_true")
    parser.add_argument("--smiles-tfidf", action="store_true")
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

    train, test = load_data()
    X_df, Xte_df, features = attach_features(
        train,
        test,
        use_smiles_basic=args.smiles_basic,
        use_smiles_tfidf=args.smiles_tfidf,
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
    X = X_df.values
    X_test = Xte_df.values
    y = train["Tm"].values

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    oof_preds, test_preds, base_cv = make_oof_predictions(
        X, y, X_test, args.folds, args.seed, models
    )

    # Meta-learner: Ridge on base model OOFs
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    # Save OOF and test preds for reproducibility
    for m in models:
        np.save(OOF_DIR / f"oof_{m}_{args.name}.npy", oof_preds[m])
        np.save(OOF_DIR / f"test_{m}_{args.name}.npy", test_preds[m])

    # Build meta features
    X_meta = np.vstack([oof_preds[m] for m in models]).T
    X_meta_test = np.vstack([test_preds[m] for m in models]).T
    meta_scaler = StandardScaler(with_mean=True)
    X_meta = meta_scaler.fit_transform(X_meta)
    X_meta_test = meta_scaler.transform(X_meta_test)

    ridge = Ridge(alpha=1.0, random_state=args.seed)
    # CV for meta
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    meta_oof = np.zeros_like(y, dtype=float)
    meta_test = np.zeros(X_test.shape[0], dtype=float)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_meta, y), 1):
        ridge.fit(X_meta[trn_idx], y[trn_idx])
        meta_oof[val_idx] = ridge.predict(X_meta[val_idx])
        meta_test += ridge.predict(X_meta_test) / args.folds
        print(f"Meta fold {fold}: {mean_absolute_error(y[val_idx], meta_oof[val_idx]):.4f}")

    meta_cv = mean_absolute_error(y, meta_oof)
    print(f"Meta CV MAE: {meta_cv:.4f} | Base: {base_cv}")

    # Save submission
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": test["id"], "Tm": meta_test})
    out_path = SUB_DIR / f"submission_{args.name}_stack_meta{meta_cv:.3f}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
