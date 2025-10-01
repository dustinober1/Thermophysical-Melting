#!/usr/bin/env python3
"""
Comprehensive experiment runner to systematically test different configurations.
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

RESULTS_LOG = Path("experiment_results.json")


def run_command(cmd, description):
    """Run a shell command and log results"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse MAE from output
    mae = None
    for line in result.stdout.split('\n'):
        if 'CV MAE:' in line:
            try:
                mae = float(line.split('CV MAE:')[1].strip())
            except:
                pass
    
    return {
        "description": description,
        "command": cmd,
        "mae": mae,
        "duration_seconds": duration,
        "timestamp": start_time.isoformat(),
        "success": result.returncode == 0,
    }


def log_result(result):
    """Append result to log file"""
    results = []
    if RESULTS_LOG.exists():
        with open(RESULTS_LOG, 'r') as f:
            results = json.load(f)
    
    results.append(result)
    
    with open(RESULTS_LOG, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Result logged to {RESULTS_LOG}")
    if result['mae']:
        print(f"  MAE: {result['mae']:.4f}")


def main():
    experiments = []
    
    # Experiment 1: Baseline with just Group features
    experiments.append({
        "description": "Baseline - Group features only (LightGBM)",
        "command": "python src/train.py --model lightgbm --name exp1_baseline"
    })
    
    # Experiment 2: Add SMILES basic features
    experiments.append({
        "description": "Group + SMILES basic features",
        "command": "python src/train.py --model lightgbm --smiles-basic --name exp2_smiles_basic"
    })
    
    # Experiment 3: Add TF-IDF with different configurations
    experiments.append({
        "description": "Group + SMILES + TF-IDF (128 components)",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name exp3_tfidf128"
    })
    
    experiments.append({
        "description": "Group + SMILES + TF-IDF (256 components, ngrams 2-5)",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 256 --tfidf-ngram-min 2 --tfidf-ngram-max 5 --name exp4_tfidf256"
    })
    
    experiments.append({
        "description": "Group + SMILES + TF-IDF (512 components, ngrams 2-6)",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 512 --tfidf-ngram-min 2 --tfidf-ngram-max 6 --name exp5_tfidf512"
    })
    
    # Experiment 6: Chemical structure features
    experiments.append({
        "description": "Group + Chemical structure features",
        "command": "python src/train.py --model lightgbm --chemical-structure --name exp6_chem_struct"
    })
    
    experiments.append({
        "description": "Group + Chemical + Advanced features",
        "command": "python src/train.py --model lightgbm --chemical-structure --advanced-features --name exp7_chem_adv"
    })
    
    experiments.append({
        "description": "Group + Chemical + Advanced + Polynomial + Interactions",
        "command": "python src/train.py --model lightgbm --chemical-structure --advanced-features --polynomial --interactions --name exp8_chem_full"
    })
    
    # Experiment 9: RDKit descriptors
    experiments.append({
        "description": "Group + RDKit descriptors",
        "command": "python src/train.py --model lightgbm --rdkit-desc --name exp9_rdkit"
    })
    
    # Experiment 10: Morgan fingerprints
    experiments.append({
        "description": "Group + Morgan fingerprints (r=2, 1024 bits)",
        "command": "python src/train.py --model lightgbm --morgan --morgan-radius 2 --morgan-nbits 1024 --name exp10_morgan"
    })
    
    # Experiment 11: MACCS keys
    experiments.append({
        "description": "Group + MACCS keys",
        "command": "python src/train.py --model lightgbm --maccs --name exp11_maccs"
    })
    
    # Experiment 12: Everything combined
    experiments.append({
        "description": "All features combined",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --chemical-structure --advanced-features --rdkit-desc --morgan --maccs --svd-components 256 --name exp12_all_features"
    })
    
    # Experiment 13: Best baseline with different models
    experiments.append({
        "description": "Best config with XGBoost",
        "command": "python src/train.py --model xgboost --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name exp13_xgb"
    })
    
    experiments.append({
        "description": "Best config with CatBoost",
        "command": "python src/train.py --model catboost --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name exp14_cat"
    })
    
    # Experiment 15: Target transformation
    experiments.append({
        "description": "Best config with log1p transform",
        "command": "python src/train_advanced.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --target-transform log1p --name exp15_log1p"
    })
    
    experiments.append({
        "description": "Best config with Yeo-Johnson transform",
        "command": "python src/train_advanced.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --target-transform yeo-johnson --name exp16_yeojohnson"
    })
    
    # Experiment 17: Sample weighting
    experiments.append({
        "description": "Best config with sample weights",
        "command": "python src/train_advanced.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --sample-weights --name exp17_weights"
    })
    
    # Experiment 18: Huber loss
    experiments.append({
        "description": "Best config with Huber loss",
        "command": "python src/train_advanced.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --huber --name exp18_huber"
    })
    
    # Experiment 19: Stacking with 2 models
    experiments.append({
        "description": "Stacking: LightGBM + CatBoost with Ridge",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost --meta-learner ridge --smiles-basic --smiles-tfidf --svd-components 128 --name exp19_stack_lgb_cat"
    })
    
    # Experiment 20: Stacking with 3 models
    experiments.append({
        "description": "Stacking: LightGBM + CatBoost + XGBoost with Ridge",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost --meta-learner ridge --smiles-basic --smiles-tfidf --svd-components 128 --name exp20_stack_3models"
    })
    
    # Experiment 21: Different meta-learners
    experiments.append({
        "description": "Stacking with ElasticNet meta-learner",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost --meta-learner elasticnet --smiles-basic --smiles-tfidf --svd-components 128 --name exp21_stack_enet"
    })
    
    experiments.append({
        "description": "Stacking with Huber meta-learner",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost --meta-learner huber --smiles-basic --smiles-tfidf --svd-components 128 --name exp22_stack_huber"
    })
    
    # Experiment 23: Stacking with more features
    experiments.append({
        "description": "Stacking with all chemical features",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost --meta-learner ridge --smiles-basic --smiles-tfidf --chemical-structure --advanced-features --svd-components 256 --name exp23_stack_chem"
    })
    
    # Experiment 24: Kitchen sink
    experiments.append({
        "description": "Kitchen sink: All features + 4 models + ElasticNet",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost,rf --meta-learner elasticnet --smiles-basic --smiles-tfidf --chemical-structure --advanced-features --rdkit-desc --morgan --svd-components 256 --name exp24_kitchen_sink"
    })
    
    print(f"\n{'#'*80}")
    print(f"RUNNING {len(experiments)} EXPERIMENTS")
    print(f"{'#'*80}\n")
    
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting: {exp['description']}")
        result = run_command(exp['command'], exp['description'])
        log_result(result)
        results.append(result)
        
        if result['mae']:
            print(f"  ✓ MAE: {result['mae']:.4f}")
        else:
            print(f"  ⚠ Could not parse MAE")
    
    # Summary
    print(f"\n{'#'*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'#'*80}\n")
    
    successful = [r for r in results if r['success'] and r['mae'] is not None]
    if successful:
        best = min(successful, key=lambda x: x['mae'])
        print(f"Best MAE: {best['mae']:.4f}")
        print(f"Description: {best['description']}")
        print(f"Command: {best['command']}")
        
        print(f"\nTop 5 experiments:")
        for i, r in enumerate(sorted(successful, key=lambda x: x['mae'])[:5], 1):
            print(f"  {i}. MAE {r['mae']:.4f}: {r['description']}")
    
    print(f"\nFull results saved to {RESULTS_LOG}")


if __name__ == "__main__":
    main()
