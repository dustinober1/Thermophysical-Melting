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

## Notes

- Current baseline uses only the 424 `Group *` numeric features and ignores SMILES.
- Next steps: add RDKit/Morgan descriptors from SMILES, hyperparameter tuning, and ensembling.
