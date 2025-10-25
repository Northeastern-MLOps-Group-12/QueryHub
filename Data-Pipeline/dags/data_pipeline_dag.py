from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

default_args = {
    'owner': 'queryhub',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_data():
    from datasets import load_dataset
    import pandas as pd
    
    dataset = load_dataset("gretelai/synthetic_text_to_sql")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    train_df.to_pickle('/tmp/train_raw.pkl')
    test_df.to_pickle('/tmp/test_raw.pkl')
    
    logging.info(f"Loaded {len(train_df)} train, {len(test_df)} test")
    return {'train': len(train_df), 'test': len(test_df)}

def validate_sql():
    import pandas as pd
    import sqlparse
    
    train_df = pd.read_pickle('/tmp/train_raw.pkl')
    test_df = pd.read_pickle('/tmp/test_raw.pkl')
    
    def is_valid(sql):
        try:
            return len(sqlparse.parse(sql)) > 0
        except:
            return False
    
    train_df = train_df[train_df['sql'].apply(is_valid)]
    test_df = test_df[test_df['sql'].apply(is_valid)]
    
    train_df.to_pickle('/tmp/train_valid.pkl')
    test_df.to_pickle('/tmp/test_valid.pkl')
    
    return {'train_valid': len(train_df), 'test_valid': len(test_df)}

def preprocess():
    import pandas as pd
    import sqlparse
    
    train_df = pd.read_pickle('/tmp/train_valid.pkl')
    test_df = pd.read_pickle('/tmp/test_valid.pkl')
    
    def format_t5(row):
        prompt = f"translate English to SQL: {row['sql_prompt']}"
        if pd.notna(row['sql_context']):
            prompt += f" context: {row['sql_context']}"
        return prompt
    
    train_df['input_text'] = train_df.apply(format_t5, axis=1)
    train_df['target_text'] = train_df['sql'].apply(lambda x: sqlparse.format(x, keyword_case='upper'))
    
    test_df['input_text'] = test_df.apply(format_t5, axis=1)
    test_df['target_text'] = test_df['sql'].apply(lambda x: sqlparse.format(x, keyword_case='upper'))
    
    train_df.to_pickle('/tmp/train_preprocessed.pkl')
    test_df.to_pickle('/tmp/test_preprocessed.pkl')
    
    return {'preprocessed': True}

def split_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    train_df = pd.read_pickle('/tmp/train_preprocessed.pkl')
    
    train_df['strat'] = train_df['domain'].astype(str) + '_' + train_df['sql_complexity'].astype(str)
    counts = train_df['strat'].value_counts()
    valid_strat = counts[counts >= 2].index
    train_df = train_df[train_df['strat'].isin(valid_strat)]
    
    train_final, val_final = train_test_split(
        train_df, test_size=0.1, stratify=train_df['strat'], random_state=42
    )
    
    train_final.to_pickle('/tmp/train_final.pkl')
    val_final.to_pickle('/tmp/val_final.pkl')
    
    return {'train': len(train_final), 'val': len(val_final)}

def save_data():
    import pandas as pd
    import json
    from datetime import datetime
    import os
    
    train = pd.read_pickle('/tmp/train_final.pkl')
    val = pd.read_pickle('/tmp/val_final.pkl')
    test = pd.read_pickle('/tmp/test_preprocessed.pkl')
    
    os.makedirs('/tmp/processed', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cols = ['input_text', 'target_text', 'domain', 'sql_complexity']
    
    for name, df in [('train', train), ('val', val), ('test', test)]:
        path = f'/tmp/processed/{name}_{timestamp}.json'
        df[cols].to_json(path, orient='records', indent=2)
    
    return {'saved': timestamp}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Gretel Text-to-SQL Pipeline',
    schedule_interval=timedelta(days=7),
    start_date=days_ago(1),
    tags=['data-pipeline'],
)

t1 = PythonOperator(task_id='load_data', python_callable=load_data, dag=dag)
t2 = PythonOperator(task_id='validate_sql', python_callable=validate_sql, dag=dag)
t3 = PythonOperator(task_id='preprocess', python_callable=preprocess, dag=dag)
t4 = PythonOperator(task_id='split_data', python_callable=split_data, dag=dag)
t5 = PythonOperator(task_id='save_data', python_callable=save_data, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5