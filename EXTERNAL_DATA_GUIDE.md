# How to Get Bradley External Datasets

The Bradley melting point datasets are critical for achieving MAE < 20. Here's how to get them:

## Option 1: Download from Kaggle Datasets

1. **Search for the dataset on Kaggle:**
   - Go to https://www.kaggle.com/datasets
   - Search for: "Bradley Melting Point Dataset" or "melting point chemical"
   - Look for datasets containing:
     - `BradleyMeltingPointDataset.xlsx`
     - `BradleyDoublePlusGoodMeltingPointDataset.xlsx`

2. **Using Kaggle API:**
   ```bash
   # Install Kaggle API if needed
   pip install kaggle
   
   # Download the dataset (replace with actual dataset path)
   kaggle datasets download -d [username]/melting-point-chemical-dataset
   
   # Unzip to data directory
   unzip melting-point-chemical-dataset.zip -d data/
   ```

## Option 2: From Research Papers

The Bradley datasets are from published research:
- **Bradley, J.-C., et al.** (2014) "Open Melting Point Datasets"
- Available from: 
  - UsefulChem project
  - GitHub: ONS-Challenge repositories
  - Figshare: https://figshare.com/

Search for:
- "Bradley melting point"
- "UsefulChem melting point dataset"
- "Open Notebook Science melting point"

## Option 3: From Winning Notebooks

Look at the winning Kaggle notebooks for this competition:
1. Go to https://www.kaggle.com/competitions/melting-point/code
2. Filter by "Best Score"
3. Find notebooks with MAE < 15
4. Check their "Data Sources" section
5. Click through to find the dataset they used

## Expected Files

You should have these files in your `data/` directory:

```
data/
├── train.csv                                    # Original competition data
├── test.csv                                     # Original competition data
├── BradleyMeltingPointDataset.xlsx             # ~50k samples
└── BradleyDoublePlusGoodMeltingPointDataset.xlsx  # ~120k samples
```

## File Format

The Bradley datasets should have these columns:
- `smiles` - SMILES string of the molecule
- `mpC` - Melting point in Celsius
- (possibly other metadata columns)

Our script will:
1. Load these files
2. Convert Celsius to Kelvin: `Tm = mpC + 273.15`
3. Merge with original training data
4. Remove duplicates

## Verify the Data

After downloading, verify:

```python
import pandas as pd

# Check file exists and can be loaded
bradley = pd.read_excel('data/BradleyMeltingPointDataset.xlsx')
print(f"Bradley dataset: {len(bradley)} rows")
print(bradley.columns.tolist())
print(bradley.head())

bradleyplus = pd.read_excel('data/BradleyDoublePlusGoodMeltingPointDataset.xlsx')
print(f"Bradley++ dataset: {len(bradleyplus)} rows")
```

Expected output:
- Bradley: ~50,000 rows
- Bradley++: ~120,000 rows
- Columns should include: `smiles`, `mpC`

## If You Can't Get External Data

You can still improve significantly with our existing approaches:
1. Comprehensive molecular fingerprints (already implemented)
2. Yeo-Johnson transformation (already implemented)
3. Feature selection (already implemented)
4. Better hyperparameter tuning
5. Model ensembling

Run without external data:
```bash
python src/train_with_external_data.py \
    --model lightgbm \
    --folds 10 \
    --name comprehensive_features
```

This will use only the competition data but with all the advanced features and techniques.

## Need Help?

If you're having trouble finding the datasets:
1. Check the Kaggle competition discussion forums
2. Look at the "External Data" discussion threads
3. Ask in the competition's Discord/Slack
4. Search for "Bradley melting point dataset download"

The datasets are publicly available as part of open science initiatives, so they should be accessible!
