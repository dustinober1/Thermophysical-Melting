# Executive Summary

Goal: Reduce MAE below 20 for melting point prediction using SMILES and Group features.

## Current Performance Snapshot

- Single-model best (CV): ~31.996 MAE
  - Model: LightGBM
  - Features: Group + SMILES basic stats + SMILES char TF-IDF (2–4) reduced with SVD (128 comps)
- Simple stacking (CV): ~31.774 MAE
  - Base: LightGBM + CatBoost (same features)
  - Meta: Ridge on OOF predictions
- Chemistry descriptors (RDKit/Morgan/MACCS) did not yet beat TF-IDF baseline in current settings.

## What We Built

- Reproducible training pipeline (`src/train.py`) with feature flags and 5-fold CV.
- Feature builder (`src/features.py`) including:
  - Group features (424 columns)
  - Basic SMILES statistics
  - Char TF-IDF + TruncatedSVD
  - RDKit/Mordred 2D descriptors, Morgan fingerprints, MACCS keys
- Stacking pipeline (`src/stacking.py`) that produces OOF predictions and trains a Ridge meta-model.
- Updated docs (`README.md`) and improved `.gitignore` and `requirements.txt`.

## Constraints & Observations

- Dataset size: ~2.6k train rows, 666 test rows — regularization and low-variance meta-models help generalization.
- TF-IDF features are strong; chemistry features require careful tuning and may benefit from different models or dimensionality reduction.
- Early stopping and KFold reproducibility are in place. GPU optional for CatBoost.

## Priority Next Actions

1) Hyperparameter Optimization (HPO) — High impact
- Add Optuna sweeps for LightGBM and CatBoost targeting CV MAE.
- Tune: num_leaves/depth, min_data_in_leaf, feature/bagging fractions, lambdas, learning_rate, catboost depth/l2/leaf_estimation.
- Budget: start with 50–100 trials; use 5-fold CV, early stop on trials with plateau.

2) Broader Stacking — Medium to high impact
- Add XGBoost as a third base model; try ElasticNet and Kernel Ridge as meta-learners.
- Consider a second-level SVD on RDKit/Morgan to compress before feeding to tree models.

3) Feature Engineering Iterations — Medium impact
- TF-IDF: expand n-grams to 2–5, test SVD dims 128–512; try sublinear_tf and different min_df.
- Chemistry: Morgan radius 2–3, bits 1024–2048; RDKit/Mordred with targeted subsets; optional PCA/SVD to 128–256 dims.
- Target transforms: test log1p or Yeo-Johnson with inverse at prediction time.

4) Robustness & Outliers — Medium impact
- Evaluate Huber/Quantile loss (where supported) and robust MAE variants.
- Winsorize or downweight extreme Tm outliers via sample_weight.

5) Caching & Speed — Developer experience
- Cache chemistry features to `data/features_*.parquet` keyed by params to reduce compute.
- Persist TF-IDF vectorizer/SVD to reuse between runs.

## Optional Experiments

- GBDT with monotonic constraints if any feature-target relationships are known.
- KRR/GPR on TF-IDF embeddings only (risk of overfit; monitor CV closely).
- Simple neural baselines on TF-IDF embeddings (MLP with dropout + early stopping).

## Milestones

- M1 (short-term): CV < 31.0 via HPO + 3-model stack (1–2 days of iteration).
- M2 (mid-term): CV < 28.0 via refined features, robust losses, and meta-learner tuning.
- M3 (stretch): Leaderboard < 20 MAE with advanced ensembling and domain-informed features.

## How to Reproduce Key Runs

- Best LGBM baseline:
  - `python src/train.py --model lightgbm --folds 5 --seed 42 --name grp_smiles_tfidf --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --tfidf-min-df 3`
- Stacking (LGBM+Cat → Ridge):
  - `python src/stacking.py --folds 5 --seed 42 --name lgb_cat_stack_tfidf128 --models lightgbm,catboost --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --tfidf-min-df 3`

