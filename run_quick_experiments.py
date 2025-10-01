#!/usr/bin/env python3
"""
Quick focused experiments to test the most promising approaches.
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime

RESULTS_LOG = Path("quick_experiment_results.json")


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
                break
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
    
    # Start with known best baseline
    print("Testing expanded feature combinations...")
    
    # Experiment 1: Baseline (known best)
    experiments.append({
        "description": "Baseline: Group + SMILES basic + TF-IDF 128",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name quick1_baseline"
    })
    
    # Experiment 2: Expand TF-IDF dimensions
    experiments.append({
        "description": "TF-IDF with 256 components and ngrams 2-5",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 256 --tfidf-ngram-min 2 --tfidf-ngram-max 5 --name quick2_tfidf256"
    })
    
    # Experiment 3: Even larger TF-IDF
    experiments.append({
        "description": "TF-IDF with 384 components and ngrams 2-6",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 384 --tfidf-ngram-min 2 --tfidf-ngram-max 6 --name quick3_tfidf384"
    })
    
    # Experiment 4: Add chemical features to baseline
    experiments.append({
        "description": "Baseline + Chemical structure + Advanced features",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --chemical-structure --advanced-features --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name quick4_chem"
    })
    
    # Experiment 5: Add RDKit to baseline
    experiments.append({
        "description": "Baseline + RDKit descriptors",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --rdkit-desc --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name quick5_rdkit"
    })
    
    # Experiment 6: Kitchen sink features
    experiments.append({
        "description": "All features combined",
        "command": "python src/train.py --model lightgbm --smiles-basic --smiles-tfidf --chemical-structure --advanced-features --rdkit-desc --morgan --maccs --svd-components 256 --name quick6_all"
    })
    
    # Experiment 7: XGBoost with best features
    experiments.append({
        "description": "XGBoost with baseline features",
        "command": "python src/train.py --model xgboost --smiles-basic --smiles-tfidf --svd-components 128 --tfidf-ngram-min 2 --tfidf-ngram-max 4 --name quick7_xgb"
    })
    
    # Experiment 8: Target transformation
    experiments.append({
        "description": "Baseline with log1p transform",
        "command": "python src/train_advanced.py --model lightgbm --smiles-basic --smiles-tfidf --svd-components 128 --target-transform log1p --name quick8_log1p"
    })
    
    # Experiment 9: Stacking 3 models
    experiments.append({
        "description": "Stacking: LightGBM + CatBoost + XGBoost",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost --meta-learner ridge --smiles-basic --smiles-tfidf --svd-components 128 --name quick9_stack3"
    })
    
    # Experiment 10: Stacking with ElasticNet
    experiments.append({
        "description": "Stacking with ElasticNet meta-learner",
        "command": "python src/stacking_advanced.py --base-models lightgbm,catboost,xgboost --meta-learner elasticnet --smiles-basic --smiles-tfidf --svd-components 128 --name quick10_stack_enet"
    })
    
    print(f"\n{'#'*80}")
    print(f"RUNNING {len(experiments)} QUICK EXPERIMENTS")
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
        print(f"✨ Best MAE: {best['mae']:.4f}")
        print(f"Description: {best['description']}")
        print(f"Command: {best['command']}")
        
        print(f"\nTop 5 experiments:")
        for i, r in enumerate(sorted(successful, key=lambda x: x['mae'])[:5], 1):
            print(f"  {i}. MAE {r['mae']:.4f}: {r['description']}")
        
        print(f"\nAll results:")
        for r in sorted(successful, key=lambda x: x['mae']):
            print(f"  MAE {r['mae']:.4f} ({r['duration_seconds']:.1f}s): {r['description']}")
    
    print(f"\n✓ Full results saved to {RESULTS_LOG}")


if __name__ == "__main__":
    main()
