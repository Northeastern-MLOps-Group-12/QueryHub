# Data Pipeline Notebooks

## Overview
Gretel Text-to-SQL dataset processing for QueryHub project.

## Notebooks
1. **1_data_exploration.ipynb** - Initial EDA on 100K samples
2. **2_data_validation.ipynb** - SQL validation & T5 preprocessing  
3. **3_data_preprocessing.ipynb** - Train/val/test splits

## Results
- Processed: 100,000 training samples
- Final splits: 89,991 train / 10,000 val / 5,851 test
- SQL validation: 100% pass rate
- Token check: 0 samples exceed T5's 512 limit

## Usage
Run notebooks sequentially in Colab or Jupyter.
