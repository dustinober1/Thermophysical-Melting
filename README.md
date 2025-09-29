# Thermophysical-Melting

A Kaggle project to predict melting point (Tm) from SMILES and group features. Ultimate goal: MAE < 20.

## Quick start

- Data: place `train.csv` and `test.csv` under `data/` (already present).
- Install deps (optional if your venv already has them):

```bash
pip install -r requirements.txt
```

- Run baseline CV and create submission (LightGBM on Group features):

```bash
python src/train.py --model lightgbm --folds 5 --seed 42 --name baseline_group
```

- Try CatBoost (CPU by default):

```bash
python src/train.py --model catboost --folds 5 --seed 42 --name baseline_group
```

Outputs: prints per-fold MAE and CV MAE, and writes `submissions/submission_<name>_<model>_cv<score>.csv`.

### Feature flags

You can add SMILES-derived features:

- `--smiles-basic`: simple text stats (length, counts, aromatic, etc.)
- `--smiles-tfidf`: char-level TF‑IDF + SVD dense features
	- `--tfidf-ngram-min`, `--tfidf-ngram-max`, `--tfidf-min-df`, `--svd-components`
- RDKit/Mordred/Morgan/MACCS from SMILES:
	- `--rdkit-desc`: Mordred 2D descriptors
	- `--morgan --morgan-radius 2 --morgan-nbits 512|1024`: Morgan fingerprints
	- `--maccs`: MACCS keys

Example (our current best baseline in CV ~32 MAE):

```bash
python src/train.py --model lightgbm --folds 5 --seed 42 \
	--name grp_smiles_tfidf \
	--smiles-basic --smiles-tfidf \
	--svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --tfidf-min-df 3
```

### Stacking (OOF meta-model)

Generate out-of-fold predictions for base models and train a simple Ridge meta-learner:

```bash
python src/stacking.py --folds 5 --seed 42 --name lgb_cat_stack \
	--smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --tfidf-min-df 3
```

This will:
- Train LightGBM and CatBoost as base learners with 5-fold OOF
- Save OOF/test preds under `oof/`
- Train Ridge on base OOFs and write `submissions/submission_<name>_stack_meta<score>.csv`

## Notes

- GPU: CatBoost can use GPU if available by setting `task_type="GPU"` in code (currently CPU default).
- OOF artifacts in `oof/` let you try alternative meta-learners (ElasticNet, KRR) quickly.
- Next steps: Optuna hyperparameter sweeps, robust losses, and expanded stacking.

## Current status

- Best single model CV: ~31.996 MAE (LightGBM + Group + SMILES-basic + TF-IDF+SVD 128 comps).
- Stacking gain: ~31.774 MAE (Ridge on LGBM + CatBoost OOFs with the same features).
- See `executive_summary.md` for a high-level summary and prioritized next actions.

## Repository layout

- `src/train.py` — single-model training with feature flags and 5-fold CV.
- `src/stacking.py` — generates base OOF preds and trains a Ridge meta-learner, saves submission.
- `src/features.py` — feature construction: Group, SMILES basic, TF-IDF+SVD, RDKit/Morgan/MACCS.
- `data/` — `train.csv` and `test.csv` live here.
- `submissions/` — generated Kaggle submissions.
- `oof/` — saved out-of-fold predictions (created by stacking pipeline).
