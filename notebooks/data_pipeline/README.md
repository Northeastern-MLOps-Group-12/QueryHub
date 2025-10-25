# Data Pipeline Notebooks

## Overview
Gretel Text-to-SQL dataset processing for QueryHub project.

## Notebooks
1. **Load&Explore.ipynb** - Data loading and initial EDA on 100K samples
2. **Validate&Prep.ipynb** - SQL validation and T5 preprocessing
3. **Split&Save.ipynb** - Train/val/test splits and JSON export

## Results
- Processed: 100,000 training samples
- Final splits: 89,991 train / 10,000 val / 5,851 test
- SQL validation: 100% pass rate
- Token check: 0 samples exceed T5's 512 limit

## Usage
Run notebooks sequentially in Colab or Jupyter.
