#!/usr/bin/env python3
"""
Clean up repository by removing unnecessary files and organizing structure.
"""
import os
import shutil
from pathlib import Path

def cleanup_repo():
    """Clean up the repository"""
    
    base_dir = Path(__file__).parent
    
    print("🧹 Cleaning up repository...")
    print("="*60)
    
    # Files/directories to remove
    items_to_remove = [
        "stack_output.log",
        ".DS_Store",
        "**/.DS_Store",  # All .DS_Store files
        "catboost_info",  # Training artifacts
        "**/__pycache__",  # Python cache
        "**/*.pyc",
        "experiment_results.json",  # If exists
        "quick_experiment_results.json",  # If exists
        "optuna_results",  # If exists and empty
    ]
    
    removed_count = 0
    
    for item in items_to_remove:
        if "**" in item:
            # Handle glob patterns
            for path in base_dir.glob(item):
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"  ✓ Removed directory: {path.relative_to(base_dir)}")
                    else:
                        path.unlink()
                        print(f"  ✓ Removed file: {path.relative_to(base_dir)}")
                    removed_count += 1
        else:
            path = base_dir / item
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  ✓ Removed directory: {path.relative_to(base_dir)}")
                else:
                    path.unlink()
                    print(f"  ✓ Removed file: {path.relative_to(base_dir)}")
                removed_count += 1
    
    # Clean OOF directory (keep directory but may want to archive old files)
    oof_dir = base_dir / "oof"
    if oof_dir.exists():
        oof_files = list(oof_dir.glob("*.npy"))
        if len(oof_files) > 10:
            print(f"\n  ℹ️  Note: {len(oof_files)} OOF prediction files in oof/")
            print(f"     Consider archiving old files if not needed")
    
    # Create necessary directories
    dirs_to_create = [
        "submissions",
        "oof",
        "optuna_results",
    ]
    
    for dir_name in dirs_to_create:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    
    print(f"\n✓ Cleanup complete! Removed {removed_count} items.")
    
    # Show final structure
    print("\n📁 Repository structure:")
    print("="*60)
    
    important_items = [
        ("📂 src/", "Core training scripts"),
        ("📂 data/", "Training and test data"),
        ("📂 submissions/", "Generated submissions"),
        ("📂 oof/", "Out-of-fold predictions"),
        ("📂 notebooks/", "Jupyter notebooks"),
        ("📄 README.md", "Main documentation"),
        ("📄 QUICK_REFERENCE.md", "Quick commands"),
        ("📄 STRATEGY_FOR_MAE_UNDER_20.md", "Winning strategy"),
        ("📄 EXTERNAL_DATA_GUIDE.md", "How to get external data"),
        ("📄 generate_submission.py", "Generate final submission"),
        ("📄 requirements.txt", "Python dependencies"),
    ]
    
    for item, description in important_items:
        item_path = item.replace("📂 ", "").replace("📄 ", "")
        if (base_dir / item_path).exists():
            print(f"  {item:30} - {description}")
    
    print("\n" + "="*60)
    print("✨ Repository is now clean and organized!")
    print("\n📝 Next steps:")
    print("  1. Run: python generate_submission.py")
    print("  2. Upload submission to Kaggle")
    print("  3. See EXTERNAL_DATA_GUIDE.md for better results")


if __name__ == "__main__":
    cleanup_repo()
