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
import xgboost as xgb
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


def kfold_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_type: str = "lightgbm",
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_STATE,
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
    # RDKit/Morgan/MACCS feature flags
    use_rdkit_desc: bool = False,
    use_morgan: bool = False,
    morgan_radius: int = 2,
    morgan_nbits: int = 1024,
    use_maccs: bool = False,
):
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

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    # Standardize for LightGBM when using leaves-wise splits (optional, CatBoost doesn't need it)
    scaler = None
    if model_type in ["lightgbm", "xgboost"]:
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
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
            )

        elif model_type == "xgboost":
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
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
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=100,
                verbose_eval=100,
            )
        if model_type == "lightgbm":
            oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

        elif model_type == "xgboost":
            dval = xgb.DMatrix(X_val)
            dtest = xgb.DMatrix(X_test)
            oof[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / n_folds

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
    parser.add_argument("--model", choices=["lightgbm", "catboost", "xgboost"], default="lightgbm")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--name", type=str, default="baseline_group_features")
    parser.add_argument("--smiles-basic", action="store_true", help="Include simple SMILES text features")
    parser.add_argument("--smiles-tfidf", action="store_true", help="Include SMILES char TF-IDF + SVD features")
    parser.add_argument("--chemical-structure", action="store_true", help="Include chemical structure features (H-bonds, symmetry, etc.)")
    parser.add_argument("--advanced-features", action="store_true", help="Include advanced chemical features (ratios, interactions)")
    parser.add_argument("--polynomial", action="store_true", help="Include polynomial features for key descriptors")
    parser.add_argument("--interactions", action="store_true", help="Include meaningful chemical interactions")
    parser.add_argument("--tfidf-ngram-min", type=int, default=2)
    parser.add_argument("--tfidf-ngram-max", type=int, default=5)
    parser.add_argument("--tfidf-min-df", type=int, default=2)
    parser.add_argument("--svd-components", type=int, default=256)
    # RDKit/Mordred features
    parser.add_argument("--rdkit-desc", action="store_true", help="Include Mordred/2D RDKit descriptors")
    parser.add_argument("--morgan", action="store_true", help="Include Morgan fingerprints")
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-nbits", type=int, default=1024)
    parser.add_argument("--maccs", action="store_true", help="Include MACCS keys")
    args = parser.parse_args()

    train, test = load_data()

    # Basic checks
    assert "Tm" in train.columns, "Target Tm missing in train.csv"
    # Build features
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
    assert len(features) > 0, "No features found"
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
        f"{n_group} Group, {n_sbasic} S-basic, {n_chem} Chem, {n_adv} Adv, {n_tfidf} TFIDF, {n_rd} RDKit, {n_mg} Morgan, {n_maccs} MACCS)"
    )
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")

    oof, preds, cv_mae = kfold_cv(
        train,
        test,
        model_type=args.model,
        n_folds=args.folds,
        seed=args.seed,
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
    )
    save_submission(test, preds, f"{args.name}_{args.model}_cv{cv_mae:.3f}")


if __name__ == "__main__":
    main()
