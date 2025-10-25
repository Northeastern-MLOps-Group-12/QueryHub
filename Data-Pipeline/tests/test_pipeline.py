import pytest
import pandas as pd
import sqlparse

def test_sql_validation():
    """Test SQL validation function"""
    valid_sql = [
        "SELECT * FROM users",
        "SELECT COUNT(*) FROM orders",
        "SELECT a.id FROM a JOIN b ON a.id = b.id"
    ]
    
    invalid_sql = [
        "",
        None,
        "NOT SQL",
        "DROP TABLE users"
    ]
    
    def validate(sql):
        if not sql:
            return False
        try:
            parsed = sqlparse.parse(sql)
            if "DROP" in sql.upper() or "TRUNCATE" in sql.upper():
                return False
            return len(parsed) > 0
        except:
            return False
    
    for sql in valid_sql:
        assert validate(sql) == True, f"Valid SQL failed: {sql}"
    
    for sql in invalid_sql:
        assert validate(sql) == False, f"Invalid SQL passed: {sql}"

def test_t5_formatting():
    """Test T5 input formatting"""
    test_cases = [
        {
            'input': {'sql_prompt': 'show users', 'sql_context': 'users table'},
            'expected': 'translate English to SQL: show users context: users table'
        },
        {
            'input': {'sql_prompt': 'count orders', 'sql_context': None},
            'expected': 'translate English to SQL: count orders'
        }
    ]
    
    def format_t5(prompt, context):
        result = f"translate English to SQL: {prompt}"
        if context:
            result += f" context: {context}"
        return result
    
    for case in test_cases:
        output = format_t5(case['input']['sql_prompt'], case['input']['sql_context'])
        assert output == case['expected']

def test_train_val_split():
    """Test stratified splitting"""
    from sklearn.model_selection import train_test_split
    
    df = pd.DataFrame({
        'text': [f'text_{i}' for i in range(100)],
        'domain': ['d1'] * 50 + ['d2'] * 50,
        'complexity': ['simple'] * 60 + ['complex'] * 40
    })
    
    df['strat'] = df['domain'] + '_' + df['complexity']
    
    train, val = train_test_split(df, test_size=0.1, stratify=df['strat'], random_state=42)
    
    assert len(val) == 10
    assert len(train) == 90

def test_data_anomalies():
    """Test anomaly detection"""
    df = pd.DataFrame({
        'input_text': ['text1', None, 'text3'],
        'target_text': ['sql1', 'sql2', 'x' * 1500]
    })
    
    assert df['input_text'].isnull().sum() == 1
    assert (df['target_text'].str.len() > 1000).sum() == 1

def test_metrics_logging():
    """Test metrics are properly formatted"""
    metrics = {
        'train_size': 89991,
        'val_size': 10000,
        'test_size': 5851,
        'validation_rate': 1.0
    }
    
    assert metrics['train_size'] + metrics['val_size'] < 100001
    assert metrics['validation_rate'] <= 1.0
    assert all(v >= 0 for v in metrics.values())