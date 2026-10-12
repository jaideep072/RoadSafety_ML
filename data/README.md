# Data Directory Guide

This directory manages dataset files separated from source code across raw and processed stages.

## Directory Structure

```text
data/
├── raw/
│   ├── US_Accidents_Sample_100k.csv     # 100k representative sample dataset (included/generated)
│   └── US_Accidents_March23.csv         # Full 7.7M records dataset (Optional, ignored by git)
└── processed/
    └── roadsafety_preprocessed.csv      # Clean, imputed, and scaled dataset for ML pipelines
```

## Large Dataset Setup

The complete `US_Accidents_March23.csv` dataset (~3.05 GB) contains 7.7 million traffic collision records from across the United States. Due to GitHub file size limits:

1. Download `US_Accidents_March23.csv` from [Kaggle US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
2. Place the downloaded CSV file into `data/raw/US_Accidents_March23.csv`.
3. If no external dataset is provided, the platform automatically utilizes `data/raw/US_Accidents_Sample_100k.csv` or synthesizes a conforming schema dataset with reproducible random seed.

## Running Preprocessing

To generate `data/processed/roadsafety_preprocessed.csv`, execute:

```bash
python scripts/preprocess_data.py
```
