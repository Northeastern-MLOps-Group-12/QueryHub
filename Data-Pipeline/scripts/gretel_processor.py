# Single script that does everything

from datasets import load_dataset
import pandas as pd
import sqlparse
import json
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

def run_pipeline():
    # 1. Load
    print("Loading data...")
    try:
        dataset = load_dataset("gretelai/synthetic_text_to_sql")
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        print(f"Loaded {len(train_df)} train, {len(test_df)} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 2. Validate SQL
    def validate_sql(sql):
        try:
            return len(sqlparse.parse(sql)) > 0
        except:
            return False
    
    train_df = train_df.dropna(subset=['sql_prompt', 'sql'])
    train_df = train_df[train_df['sql'].apply(validate_sql)]
    test_df = test_df[test_df['sql'].apply(validate_sql)]
    
    # 3. Format for T5
    def format_for_t5(row):
        if pd.notna(row['sql_context']) and row['sql_context'].strip():
            return f"translate English to SQL: {row['sql_prompt']} context: {row['sql_context']}"
        return f"translate English to SQL: {row['sql_prompt']}"
    
    train_df['input_text'] = train_df.apply(format_for_t5, axis=1)
    train_df['target_text'] = train_df['sql'].apply(lambda x: sqlparse.format(x, keyword_case='upper').strip() if x else x)
    test_df['input_text'] = test_df.apply(format_for_t5, axis=1)
    test_df['target_text'] = test_df['sql'].apply(lambda x: sqlparse.format(x, keyword_case='upper').strip() if x else x)
    
    # 4. Create splits
    train_df['strat_key'] = train_df['domain'].astype(str) + '_' + train_df['sql_complexity'].astype(str)
    valid_keys = train_df['strat_key'].value_counts()
    valid_keys = valid_keys[valid_keys >= 2].index
    train_df = train_df[train_df['strat_key'].isin(valid_keys)]
    
    train_final, val_final = train_test_split(train_df, test_size=0.1, stratify=train_df['strat_key'], random_state=42)
    
    # 5. Clean and save
    cols = ['input_text', 'target_text', 'domain', 'sql_complexity']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    datasets = {
        'train': train_final[cols].to_dict('records'),
        'val': val_final[cols].to_dict('records'),
        'test': test_df[cols].to_dict('records')
    }
    
    for name, data in datasets.items():
        filepath = f'data/processed/{name}_{timestamp}.json'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"✓ Complete - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    print(f"✓ Files saved in data/processed/ with timestamp: {timestamp}")

if __name__ == "__main__":
    run_pipeline()