#!/usr/bin/env python3
"""
Generate final submission using best approach without external data.
Uses comprehensive features + stacking for best CV performance.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stacking_advanced import load_data, generate_base_predictions, train_meta_model
from features import attach_features

RANDOM_STATE = 42
N_FOLDS = 5


def main():
    print("="*80)
    print("GENERATING FINAL SUBMISSION")
    print("="*80)
    print("\nStrategy: 3-Model Stacking (LightGBM + CatBoost + XGBoost)")
    print("Features: Group + SMILES + TF-IDF 256 + Chemical Structure")
    print("Meta-learner: Ridge regression")
    print("CV Folds: 5")
    
    # Load data
    print("\n[1/4] Loading data...")
    train, test = load_data()
    
    # Build features
    print("[2/4] Building features...")
    X_df, Xte_df, features = attach_features(
        train,
        test,
        use_smiles_basic=True,
        use_smiles_tfidf=True,
        use_chemical_structure=True,
        use_advanced_features=True,
        tfidf_ngram_min=2,
        tfidf_ngram_max=5,
        svd_components=256,
        random_state=RANDOM_STATE,
    )
    
    print(f"Total features: {len(features)}")
    print(f"  - Group: {sum(c.startswith('Group') for c in features)}")
    print(f"  - SMILES basic: {sum(c.startswith('S_') for c in features)}")
    print(f"  - Chemical: {sum(c.startswith('CHEM_') for c in features)}")
    print(f"  - Advanced: {sum(c.startswith('ADV_') for c in features)}")
    print(f"  - TF-IDF: {sum(c.startswith('TFIDF_SVD_') for c in features)}")
    
    X = X_df.values
    y = train["Tm"].values
    X_test = Xte_df.values
    
    # Generate base model predictions
    print("\n[3/4] Training base models (this may take 10-15 minutes)...")
    base_models = ['lightgbm', 'catboost', 'xgboost']
    
    oof_preds, test_preds = generate_base_predictions(
        base_models=base_models,
        X=X,
        y=y,
        X_test=X_test,
        n_folds=N_FOLDS,
        seed=RANDOM_STATE,
        custom_params=None,
    )
    
    # Train meta-model
    print("\n[4/4] Training meta-model...")
    meta_model = train_meta_model('ridge', oof_preds, y, alpha=1.0)
    final_preds = meta_model.predict(test_preds)
    
    # Calculate CV score
    oof_meta = meta_model.predict(oof_preds)
    from sklearn.metrics import mean_absolute_error
    cv_mae = mean_absolute_error(y, oof_meta)
    
    print(f"\n{'='*80}")
    print(f"Final CV MAE: {cv_mae:.4f}")
    print(f"{'='*80}")
    
    # Save submission
    submission_dir = Path(__file__).parent / "submissions"
    submission_dir.mkdir(exist_ok=True)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'Tm': final_preds
    })
    
    submission_path = submission_dir / f"final_submission_3model_stack_cv{cv_mae:.3f}.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission saved to: {submission_path}")
    print(f"✓ Ready for Kaggle upload!")
    
    # Also save simple average for comparison
    avg_preds = test_preds.mean(axis=1)
    oof_avg = oof_preds.mean(axis=1)
    avg_mae = mean_absolute_error(y, oof_avg)
    
    avg_submission = pd.DataFrame({
        'id': test['id'],
        'Tm': avg_preds
    })
    avg_path = submission_dir / f"simple_average_cv{avg_mae:.3f}.csv"
    avg_submission.to_csv(avg_path, index=False)
    print(f"\n✓ Simple average also saved to: {avg_path}")
    print(f"  (CV MAE: {avg_mae:.4f})")
    
    print("\n" + "="*80)
    print("SUBMISSION GENERATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Upload the submission file to Kaggle")
    print("2. For better results, obtain Bradley external datasets")
    print("   (See EXTERNAL_DATA_GUIDE.md)")
    print("3. Run with external data for MAE < 20 (potentially < 10)")


if __name__ == "__main__":
    main()
