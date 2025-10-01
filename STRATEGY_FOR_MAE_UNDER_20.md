# Strategies to Achieve MAE < 20 (and potentially < 10)

Based on analysis of top-performing Kaggle notebooks, here are the key strategies:

## 🎯 Current Status
- **Best CV MAE**: ~31.5 (3-model stacking with Ridge)
- **Target**: < 20 MAE (stretch goal: < 10)

## 🔑 Critical Success Factors from Top Notebooks

### 1. **External Data (HUGE Impact)**
The winning notebooks use Bradley melting point datasets:
- **BradleyMeltingPointDataset.xlsx**: ~50k samples
- **BradleyDoublePlusGoodMeltingPointDataset.xlsx**: ~120k samples
- Combined with original train: **~170k+ training samples** vs our 2.6k

**Action**: Download from Kaggle dataset: `melting-point-chemical-dataset`
```bash
kaggle datasets download -d YOUR_USERNAME/melting-point-chemical-dataset
```

### 2. **Comprehensive Molecular Fingerprints (6900+ features)**
Winners extract ALL available fingerprints:
- ✅ RDKit Descriptors (217 features)
- ✅ Morgan/ECFP (1024 bits)
- ✅ Functional-Class FP/FCFP (1024 bits)  
- ✅ MACCS Keys (167 bits)
- ✅ AtomPair (2048 bits)
- ✅ RDKit FP (2048 bits)
- ✅ Avalon FP (1024 bits)

**Total**: ~6,500 features

### 3. **Yeo-Johnson Target Transformation (Critical)**
```python
from sklearn.preprocessing import PowerTransformer
yeo = PowerTransformer(method='yeo-johnson')
y_train_transformed = yeo.fit_transform(y_train.reshape(-1, 1))
# ... train model ...
predictions = yeo.inverse_transform(predictions.reshape(-1, 1))
```
This normalizes the target distribution for better model performance.

### 4. **Feature Selection Inside CV Loop**
```python
# Build small model to identify important features
selector_model = LGBMRegressor(n_estimators=500, max_depth=6)
selector_model.fit(X_train, y_train)
selector = SelectFromModel(selector_model, threshold="mean")
X_train_selected = selector.transform(X_train)
```
Reduces overfitting by keeping only relevant features per fold.

### 5. **10-Fold Cross-Validation**
More folds = better generalization, especially with larger datasets.

### 6. **Simple/Default Hyperparameters**
Winners use simple hyperparameters after Optuna tuning:
```python
# LightGBM
params = {
    'objective': 'huber',  # or 'mae'
    'learning_rate': 0.05-0.1,
    'num_leaves': 31-32,
    'max_depth': 6,
    'n_estimators': 15000,
    'early_stopping_rounds': 200
}
```

## 📊 Implementation Priority

### High Priority (Do First)
1. ✅ **Implement Yeo-Johnson transformation** - Already in `train_with_external_data.py`
2. ✅ **Implement feature selection** - Already in `train_with_external_data.py`
3. ✅ **Add all fingerprint types** - Already in `train_with_external_data.py`
4. ⏳ **Download Bradley external datasets**
5. ⏳ **Run 10-fold CV with external data**

### Medium Priority
1. Test different model combinations with external data
2. Try different feature selection thresholds
3. Ensemble multiple runs

### Lower Priority (Diminishing Returns)
1. Extensive hyperparameter tuning (winners use simple params)
2. Complex stacking architectures
3. Additional feature engineering on Group features

## 🚀 Quick Start Guide

### Step 1: Get External Data
```bash
# Download Bradley datasets (need Kaggle account)
mkdir -p data
cd data
# Manual download from: 
# https://www.kaggle.com/datasets/.../melting-point-chemical-dataset
# Or use Kaggle API if available
```

### Step 2: Run with External Data
```bash
python src/train_with_external_data.py \
    --model lightgbm \
    --folds 10 \
    --use-external \
    --name external_lgbm_v1
```

### Step 3: Try Different Models
```bash
# XGBoost
python src/train_with_external_data.py \
    --model xgboost \
    --folds 10 \
    --use-external \
    --name external_xgb_v1

# CatBoost  
python src/train_with_external_data.py \
    --model catboost \
    --folds 10 \
    --use-external \
    --name external_cat_v1
```

### Step 4: Ensemble
Average predictions from multiple models trained on external data.

## 📈 Expected Results

Based on winning notebooks:
- **Without external data**: MAE ~25-30 (our current: 31.5)
- **With external data**: MAE ~10-17 (winners achieved 8-12)

The external data is the **biggest single improvement** you can make.

## 🛠️ Files Created

1. `src/train_with_external_data.py` - Complete implementation with:
   - External data loading
   - All fingerprint types
   - Yeo-Johnson transformation
   - Feature selection
   - 10-fold CV

2. `src/optuna_hpo.py` - Hyperparameter optimization (optional)

3. `src/train_advanced.py` - Advanced training with target transforms

4. `src/stacking_advanced.py` - Multi-model stacking

## 📝 Notes

- The external datasets contain ~170k molecules with melting points
- Combining them with original train data provides massive regularization
- Feature selection prevents overfitting despite 6900+ features
- Yeo-Johnson makes the target more Gaussian-like for tree models
- Simple hyperparameters work better than over-tuned ones with large datasets

## 🎯 Bottom Line

To achieve MAE < 20:
1. **Must have**: External Bradley datasets
2. **Must have**: Yeo-Johnson transformation  
3. **Must have**: Comprehensive fingerprints
4. **Nice to have**: Feature selection, 10-fold CV, model ensembling

The external data alone should drop your MAE from ~31 to ~20-25.
Combined with other techniques, you can reach < 15 or even < 10.
