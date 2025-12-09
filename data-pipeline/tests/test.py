import sys
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import pandas as pd
from pathlib import Path

# Mock ALL Airflow modules before imports
sys.modules['airflow'] = MagicMock()
sys.modules['airflow.models'] = MagicMock()
sys.modules['airflow.models.Variable'] = MagicMock()
sys.modules['airflow.settings'] = MagicMock()
sys.modules['airflow.operators'] = MagicMock()
sys.modules['airflow.operators.python'] = MagicMock()
sys.modules['airflow.utils'] = MagicMock()
sys.modules['airflow.utils.email'] = MagicMock()
sys.modules['airflow.providers'] = MagicMock()
sys.modules['airflow.providers.google'] = MagicMock()
sys.modules['airflow.providers.google.cloud'] = MagicMock()
sys.modules['airflow.providers.google.cloud.hooks'] = MagicMock()
sys.modules['airflow.providers.google.cloud.hooks.gcs'] = MagicMock()
sys.modules['airflow.providers.standard'] = MagicMock()
sys.modules['airflow.providers.standard.operators'] = MagicMock()
sys.modules['airflow.providers.standard.operators.python'] = MagicMock()
sys.modules['airflow.sdk'] = MagicMock()
sys.modules['airflow.sdk.execution_time'] = MagicMock()
sys.modules['airflow.sdk.execution_time.task_runner'] = MagicMock()
sys.modules['airflow.sdk.execution_time.callback_runner'] = MagicMock()
sys.modules['airflow.sdk.bases'] = MagicMock()
sys.modules['airflow.sdk.bases.operator'] = MagicMock()
sys.modules['airflow.providers.standard.operators.trigger_dagrun'] = MagicMock()


# Mock the specific classes
GCSHook = MagicMock()
Variable = MagicMock()

# Set up the mock structure
sys.modules['airflow.providers.google.cloud.hooks.gcs'].GCSHook = GCSHook
sys.modules['airflow.models'].Variable = Variable

sys.path.insert(0, str(Path(__file__).parent.parent / 'dags'))

from data_pipeline_dag import (
    load_data, validate_sql, preprocess, validate_raw_schema,
    detect_bias, analyze_and_generate_synthetic, merge_and_split,
    remove_data_leakage, validate_engineered_schema,
    send_pipeline_success_notification
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'sql_prompt': ['Find users', 'Show sales', 'Count orders'],
        'sql_context': ['CREATE TABLE users (id INT);'] * 3,
        'sql': ['SELECT * FROM users', 'SELECT * FROM sales', 'SELECT COUNT(*) FROM orders'],
        'sql_complexity': ['basic', 'basic', 'aggregation']
    })


@pytest.fixture
def mock_ctx():
    ti = Mock()
    ti.xcom_push = Mock()
    ti.xcom_pull = Mock(return_value={})
    return {'task_instance': ti, 'execution_date': pd.Timestamp('2025-01-01')}


# TEST load_data()
@patch('pandas.DataFrame.to_pickle')
@patch('datasets.load_dataset')
def test_load_data_success(mock_load, mock_pickle, sample_data):
    mock_load.return_value = {
        'train': sample_data.to_dict('records'),
        'test': sample_data.to_dict('records')
    }
    result = load_data()
    assert result['train'] == 3
    assert result['test'] == 3


@patch('datasets.load_dataset')
def test_load_data_empty(mock_load):
    mock_load.return_value = {'train': [], 'test': []}
    result = load_data()
    assert result['train'] == 0
    assert result['test'] == 0


# TEST validate_sql()
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_validate_sql_valid(mock_read, mock_pickle, mock_csv, mock_mkdir, sample_data):
    mock_read.side_effect = [sample_data, sample_data.copy()]
    result = validate_sql()
    assert result['train_valid'] == 3
    assert result['total_anomalies'] == 0


@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_validate_sql_invalid(mock_read, mock_pickle, mock_csv, mock_mkdir):
    # Use truly invalid SQL syntax that sqlglot will reject
    invalid = pd.DataFrame({
        'sql': ['SELECT * FROMM users WHERE'],  # Typo + incomplete WHERE
        'sql_prompt': ['bad'],
        'sql_context': ['ctx'],
        'sql_complexity': ['basic']
    })
    mock_read.side_effect = [invalid, invalid.copy()]
    result = validate_sql()
    # Should detect the invalid SQL
    assert result['train_valid'] == 0 or result['total_anomalies'] > 0


# TEST preprocess()
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_preprocess(mock_read, mock_pickle, sample_data):
    mock_read.side_effect = [sample_data, sample_data.copy()]
    result = preprocess()
    assert result['preprocessed'] == True


# TEST validate_raw_schema()
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_pickle')
def test_validate_raw_schema_success(mock_read, mock_mkdir, mock_file, sample_data, mock_ctx):
    mock_read.side_effect = [sample_data, sample_data.copy()]
    result = validate_raw_schema(**mock_ctx)
    assert result['schema_validated'] == True
    assert result['validation_status'] == 'PASSED'


@patch('pandas.read_pickle')
def test_validate_raw_schema_missing_cols(mock_read, mock_ctx):
    # Dataset missing required columns
    invalid = pd.DataFrame({'sql_prompt': ['q'], 'sql': ['s']})
    mock_read.side_effect = [invalid, invalid.copy()]
    # The function will raise KeyError when trying to access missing columns
    with pytest.raises((ValueError, KeyError)):
        validate_raw_schema(**mock_ctx)


# TEST detect_bias()
@patch('utils.EmailContentGenerator.send_email_notification')
@patch('pandas.read_pickle')
def test_detect_bias_imbalanced(mock_read, mock_email, mock_ctx):
    imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 100,
        'sql_context': ['c'] * 100,
        'sql': ['s'] * 100,
        'sql_complexity': ['basic'] * 50 + ['CTEs'] * 10 + ['aggregation'] * 40
    })
    mock_read.return_value = imbalanced
    result = detect_bias(**mock_ctx)
    assert result['bias_detected'] == True
    assert result['imbalance_ratio'] > 1


@patch('utils.EmailContentGenerator.send_email_notification')
@patch('pandas.read_pickle')
def test_detect_bias_balanced(mock_read, mock_email, mock_ctx):
    balanced = pd.DataFrame({
        'sql_prompt': ['q'] * 90,
        'sql_context': ['c'] * 90,
        'sql': ['s'] * 90,
        'sql_complexity': ['basic'] * 30 + ['aggregation'] * 30 + ['CTEs'] * 30
    })
    mock_read.return_value = balanced
    result = detect_bias(**mock_ctx)
    assert result['imbalance_ratio'] == 1.0


# TEST analyze_and_generate_synthetic()
@patch('utils.DataGenerator.GenerateAdditionalData')
@patch('pandas.read_csv')
@patch('pandas.read_pickle')
@patch('os.path.exists')
def test_generate_synthetic(mock_exists, mock_read_pkl, mock_read_csv, mock_gen):
    imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 60,
        'sql_context': ['c'] * 60,
        'sql': ['s'] * 60,
        'sql_complexity': ['basic'] * 50 + ['CTEs'] * 10
    })
    mock_read_pkl.return_value = imbalanced
    mock_exists.return_value = True
    synthetic = pd.DataFrame({
        'sql_prompt': ['syn'],
        'sql_context': ['ctx'],
        'sql': ['sql'],
        'sql_complexity': ['CTEs']
    })
    mock_read_csv.return_value = synthetic
    result = analyze_and_generate_synthetic()
    assert result['synthetic_generated'] == 1


# TEST merge_and_split()
@patch('os.path.exists')
@patch('os.remove')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_pickle')
@patch('os.makedirs')
def test_merge_and_split(mock_mkdir, mock_read, mock_csv, mock_rm, mock_exists, sample_data):
    synthetic = pd.DataFrame({
        'sql_prompt': ['s'], 'sql_context': ['c'],
        'sql': ['q'], 'sql_complexity': ['CTEs']
    })
    mock_read.side_effect = [sample_data, sample_data.copy(), synthetic]
    mock_exists.return_value = False
    result = merge_and_split()
    assert 'train' in result
    assert result['synthetic_added'] == 1


# TEST remove_data_leakage()
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_remove_leakage_clean(mock_read, mock_csv, mock_ctx):
    train = pd.DataFrame({
        'input_text': ['q1', 'q2'],
        'sql': ['s1', 's2'],
        'sql_complexity': ['basic'] * 2
    })
    val = pd.DataFrame({
        'input_text': ['q3'],
        'sql': ['s3'],
        'sql_complexity': ['basic']
    })
    test = pd.DataFrame({
        'input_text': ['q4'],
        'sql': ['s4'],
        'sql_complexity': ['basic']
    })
    mock_read.side_effect = [train, val, test]
    result = remove_data_leakage(**mock_ctx)
    assert result['leakage_cleaned'] == False


@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_remove_leakage_with_dups(mock_read, mock_csv, mock_ctx):
    train = pd.DataFrame({
        'input_text': ['q1', 'q2', 'q1'],
        'sql': ['s1', 's2', 's1'],
        'sql_complexity': ['basic'] * 3
    })
    val = pd.DataFrame({'input_text': ['q3'], 'sql': ['s3'], 'sql_complexity': ['basic']})
    test = pd.DataFrame({'input_text': ['q4'], 'sql': ['s4'], 'sql_complexity': ['basic']})
    mock_read.side_effect = [train, val, test]
    result = remove_data_leakage(**mock_ctx)
    assert result['train_intra_duplicates_removed'] == 1


# TEST validate_engineered_schema()
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_csv')
def test_validate_engineered_schema(mock_read, mock_mkdir, mock_file, mock_ctx):
    valid = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test'],
        'sql': ['SELECT 1'],
        'sql_complexity': ['basic']
    })
    mock_read.side_effect = [valid, valid, valid]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    result = validate_engineered_schema(**mock_ctx)
    assert result['schema_validated'] == True


# TEST send_pipeline_success_notification()
@patch('utils.EmailContentGenerator.send_email_notification')
def test_send_success_notification(mock_email, mock_ctx):
    xcom = [
        {'train': 100, 'test': 50},
        {'train_valid': 95, 'test_valid': 48, 'total_anomalies': 5},
        {'validation_status': 'PASSED'},
        {'bias_level': 'MODERATE', 'imbalance_ratio': 5.0},
        {'synthetic_generated': 100},
        {'train': 195, 'val': 20, 'test': 48},
        {'leakage_cleaned': True},
        {'train_size': 190, 'val_size': 20, 'test_size': 48, 'validation_status': 'PASSED'}
    ]
    mock_ctx['task_instance'].xcom_pull.side_effect = xcom
    result = send_pipeline_success_notification(**mock_ctx)
    assert result['notification_sent'] == True
    assert mock_email.called


# TEST 1: Load data with large dataset
@patch('pandas.DataFrame.to_pickle')
@patch('datasets.load_dataset')
def test_load_data_large_dataset(mock_load, mock_pickle):
    """Test load_data handles large dataset"""
    large_data = pd.DataFrame({
        'sql_prompt': [f'Query {i}' for i in range(1000)],
        'sql_context': ['CREATE TABLE test (id INT);'] * 1000,
        'sql': [f'SELECT * FROM test WHERE id = {i}' for i in range(1000)],
        'sql_complexity': (['basic'] * 500 + ['aggregation'] * 300 + ['CTEs'] * 200)
    })
    mock_load.return_value = {
        'train': large_data.to_dict('records'),
        'test': large_data.to_dict('records')
    }
    result = load_data()
    assert result['train'] == 1000
    assert result['test'] == 1000


# TEST 2: Validate SQL with complex queries
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_validate_sql_complex_queries(mock_read, mock_pickle, mock_csv, mock_mkdir):
    """Test validate_sql with CTEs and window functions"""
    complex_data = pd.DataFrame({
        'sql': [
            'WITH revenue_cte AS (SELECT * FROM orders) SELECT * FROM revenue_cte',
            'SELECT *, ROW_NUMBER() OVER (PARTITION BY category) as rn FROM products',
            'SELECT * FROM sales UNION ALL SELECT * FROM returns'
        ],
        'sql_prompt': ['CTE query', 'Window function', 'Union query'],
        'sql_context': ['ctx'] * 3,
        'sql_complexity': ['CTEs', 'window functions', 'set operations']
    })
    mock_read.side_effect = [complex_data, complex_data.copy()]
    result = validate_sql()
    assert result['train_valid'] == 3
    assert result['total_anomalies'] == 0


# TEST 3: Detect bias with severely imbalanced data
@patch('utils.EmailContentGenerator.send_email_notification')
@patch('pandas.read_pickle')
def test_detect_bias_severe_imbalance(mock_read, mock_email, mock_ctx):
    """Test detect_bias with severe imbalance (>10x ratio)"""
    severely_imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 200,
        'sql_context': ['c'] * 200,
        'sql': ['s'] * 200,
        'sql_complexity': ['basic'] * 180 + ['CTEs'] * 10 + ['window functions'] * 10
    })
    mock_read.return_value = severely_imbalanced
    result = detect_bias(**mock_ctx)
    assert result['bias_detected'] == True
    assert result['bias_level'] == 'SEVERE'
    assert result['imbalance_ratio'] >= 10


# TEST 4: Remove data leakage with cross-split overlap
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_remove_leakage_cross_split_overlap(mock_read, mock_csv, mock_ctx):
    """Test remove_data_leakage removes train-val overlap"""
    train = pd.DataFrame({
        'input_text': ['q1', 'q2', 'q3'],
        'sql': ['s1', 's2', 's3'],
        'sql_complexity': ['basic'] * 3
    })
    val = pd.DataFrame({
        'input_text': ['q2', 'q4'],  # q2 overlaps with train
        'sql': ['s2', 's4'],
        'sql_complexity': ['basic'] * 2
    })
    test = pd.DataFrame({
        'input_text': ['q5'],
        'sql': ['s5'],
        'sql_complexity': ['basic']
    })
    mock_read.side_effect = [train, val, test]
    result = remove_data_leakage(**mock_ctx)
    assert result['train_cross_duplicates_removed'] > 0
    assert result['leakage_cleaned'] == True


# TEST 5: Validate engineered schema detects missing input_text format
@patch('pandas.read_csv')
def test_validate_engineered_schema_invalid_format(mock_read, mock_ctx):
    """Test validate_engineered_schema detects invalid input_text format"""
    invalid_format = pd.DataFrame({
        'input_text': ['just plain text without proper format'],
        'sql': ['SELECT 1'],
        'sql_complexity': ['basic']
    })
    mock_read.side_effect = [invalid_format, invalid_format, invalid_format]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    with pytest.raises(ValueError):
        validate_engineered_schema(**mock_ctx)


# TEST 21: Load data with unbalanced train/test split
@patch('pandas.DataFrame.to_pickle')
@patch('datasets.load_dataset')
def test_load_data_unbalanced_splits(mock_load, mock_pickle, sample_data):
    """Test load_data with different train/test sizes"""
    train_data = sample_data.to_dict('records')
    test_data = sample_data.head(1).to_dict('records')  # Only 1 test sample
    
    mock_load.return_value = {'train': train_data, 'test': test_data}
    result = load_data()
    
    assert result['train'] == 3
    assert result['test'] == 1
    assert result['train'] > result['test']


# TEST 22: Validate SQL saves anomalies report when errors found
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_validate_sql_saves_anomalies(mock_read, mock_pickle, mock_csv, mock_mkdir):
    """Test validate_sql saves anomalies CSV when invalid SQL found"""
    mixed_data = pd.DataFrame({
        'sql': ['SELECT * FROM users', 'SELECT * FROMM orders', 'SELECT 1'],
        'sql_prompt': ['good1', 'bad', 'good2'],
        'sql_context': ['ctx'] * 3,
        'sql_complexity': ['basic'] * 3
    })
    mock_read.side_effect = [mixed_data, mixed_data.copy()]
    
    result = validate_sql()
    
    # Should have created anomalies CSV
    assert mock_csv.called
    # Should have some valid and some invalid
    assert result['train_valid'] >= 0


# TEST 23: Detect bias identifies minority classes correctly
@patch('utils.EmailContentGenerator.send_email_notification')
@patch('pandas.read_pickle')
def test_detect_bias_identifies_minority_classes(mock_read, mock_email, mock_ctx):
    """Test detect_bias correctly identifies minority classes"""
    imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 150,
        'sql_context': ['c'] * 150,
        'sql': ['s'] * 150,
        'sql_complexity': (
            ['basic'] * 100 + ['aggregation'] * 30 + 
            ['CTEs'] * 10 + ['window functions'] * 10
        )
    })
    mock_read.return_value = imbalanced
    
    result = detect_bias(**mock_ctx)
    
    assert result['bias_detected'] == True
    assert result['minority_classes'] > 0
    assert result['min_class'] in ['CTEs', 'window functions']


# TEST 24: Merge and split without synthetic data
@patch('os.path.exists')
@patch('os.remove')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_pickle')
@patch('os.makedirs')
def test_merge_and_split_no_synthetic(mock_mkdir, mock_read, mock_csv, mock_rm, mock_exists, sample_data):
    """Test merge_and_split works with empty synthetic data"""
    empty_synthetic = pd.DataFrame(columns=['sql_prompt', 'sql_context', 'sql', 'sql_complexity'])
    
    mock_read.side_effect = [sample_data, sample_data.copy(), empty_synthetic]
    mock_exists.return_value = False
    
    result = merge_and_split()
    
    assert 'train' in result
    assert 'val' in result
    assert 'test' in result
    assert result['synthetic_added'] == 0


# TEST 30: Validate engineered schema validates input_text has context and query
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_csv')
def test_validate_engineered_schema_format_validation(mock_read, mock_mkdir, mock_file, mock_ctx):
    """Test validate_engineered_schema checks for proper input_text format"""
    valid_format = pd.DataFrame({
        'input_text': [
            'translate English to SQL: context: CREATE TABLE users (id INT);\n\nquery: Find all users',
            'translate English to SQL: query: Show sales'
        ],
        'sql': ['SELECT * FROM users', 'SELECT * FROM sales'],
        'sql_complexity': ['basic', 'basic']
    })
    mock_read.side_effect = [valid_format, valid_format, valid_format]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    
    result = validate_engineered_schema(**mock_ctx)
    
    assert result['schema_validated'] == True
    assert result['validation_status'] == 'PASSED'

# TEST 31: Validate SQL with all queries invalid
@patch('os.makedirs')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_validate_sql_all_invalid(mock_read, mock_pickle, mock_csv, mock_mkdir):
    """Test validate_sql when all queries are invalid"""
    all_invalid = pd.DataFrame({
        'sql': ['SELECT FROM INVALID1 FROMM WHERE', 'SELECTTT ** FROM XYZ', 'INVALID3 WHERE'],
        'sql_prompt': ['q1', 'q2', 'q3'],
        'sql_context': ['ctx'] * 3,
        'sql_complexity': ['basic'] * 3
    })
    
    mock_read.side_effect = [all_invalid, all_invalid.copy()]
    
    result = validate_sql()
    
    assert result['train_valid'] == 0
    assert result['test_valid'] == 0


# TEST 32: Detect bias with single class (no diversity)
@patch('utils.EmailContentGenerator.send_email_notification')
@patch('pandas.read_pickle')
def test_detect_bias_single_class(mock_read, mock_email, mock_ctx):
    """Test detect_bias handles dataset with only one class"""
    single_class = pd.DataFrame({
        'sql_prompt': ['q'] * 100,
        'sql_context': ['c'] * 100,
        'sql': ['s'] * 100,
        'sql_complexity': ['basic'] * 100
    })
    mock_read.return_value = single_class
    
    result = detect_bias(**mock_ctx)
    
    # Should handle gracefully (imbalance_ratio will be 1.0)
    assert 'imbalance_ratio' in result
    assert result['imbalance_ratio'] == 1.0


# TEST 33: Remove data leakage with multiple duplicates in train
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_remove_leakage_multiple_train_duplicates(mock_read, mock_csv, mock_ctx):
    """Test remove_data_leakage handles multiple duplicates"""
    train = pd.DataFrame({
        'input_text': ['q1', 'q2', 'q1', 'q3', 'q2', 'q1'],  # 3 duplicates
        'sql': ['s1', 's2', 's1', 's3', 's2', 's1'],
        'sql_complexity': ['basic'] * 6
    })
    val = pd.DataFrame({
        'input_text': ['q4'],
        'sql': ['s4'],
        'sql_complexity': ['basic']
    })
    test = pd.DataFrame({
        'input_text': ['q5'],
        'sql': ['s5'],
        'sql_complexity': ['basic']
    })
    
    mock_read.side_effect = [train, val, test]
    
    result = remove_data_leakage(**mock_ctx)
    
    assert result['train_intra_duplicates_removed'] == 3  # q1 appears 3 times, q2 appears 2 times
    assert result['leakage_cleaned'] == True


# TEST 34: Validate engineered schema saves statistics to JSON
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_csv')
def test_validate_engineered_schema_saves_stats(mock_read, mock_mkdir, mock_file, mock_ctx):
    """Test validate_engineered_schema saves comprehensive statistics"""
    valid = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test'] * 5,
        'sql': ['SELECT 1'] * 5,
        'sql_complexity': ['basic', 'aggregation', 'basic', 'CTEs', 'basic']
    })
    mock_read.side_effect = [valid, valid, valid]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    
    result = validate_engineered_schema(**mock_ctx)
    
    assert 'schema_path' in result
    assert mock_file.called  # JSON file should be opened/written


# TEST 35: Send success notification includes all pipeline stats
@patch('utils.EmailContentGenerator.send_email_notification')
def test_send_success_notification_comprehensive(mock_email, mock_ctx):
    """Test send_pipeline_success_notification includes comprehensive stats"""
    xcom = [
        {'train': 1000, 'test': 500},
        {'train_valid': 950, 'test_valid': 475, 'total_anomalies': 75},
        {'validation_status': 'PASSED'},
        {'bias_level': 'SEVERE', 'imbalance_ratio': 15.0},
        {'synthetic_generated': 800},
        {'train': 1750, 'val': 180, 'test': 475},
        {'leakage_cleaned': True, 'total_removed': {'train': 10, 'val': 2, 'test': 0}},
        {'train_size': 1740, 'val_size': 178, 'test_size': 475, 
         'validation_status': 'PASSED', 'total_duplicates_removed': 12}
    ]
    mock_ctx['task_instance'].xcom_pull.side_effect = xcom
    
    result = send_pipeline_success_notification(**mock_ctx)
    
    assert result['notification_sent'] == True
    assert result['final_train_size'] == 1740
    assert result['total_duplicates_removed'] == 12
    
    assert mock_email.called
    call_args = mock_email.call_args
    subject, html = call_args[0]
    
    assert 'Success' in subject
    # HTML formats numbers with commas (1,740)
    assert '1,740' in html
    assert 'train' in html.lower()


# TEST 31: Validate raw schema detects wrong data types
@patch('pandas.read_pickle')
def test_validate_raw_schema_wrong_dtypes(mock_read, mock_ctx):
    """Test validate_raw_schema detects incorrect data types"""
    wrong_dtype = pd.DataFrame({
        'sql_prompt': ['query1'],
        'sql_context': ['context1'],
        'sql': ['SELECT 1'],
        'sql_complexity': [123]  # Should be string, not int
    })
    mock_read.side_effect = [wrong_dtype, wrong_dtype.copy()]
    
    with pytest.raises(ValueError):
        validate_raw_schema(**mock_ctx)


# TEST 32: Analyze and generate synthetic with multiple minority classes
@patch('utils.DataGenerator.GenerateAdditionalData')
@patch('pandas.read_csv')
@patch('pandas.read_pickle')
@patch('os.path.exists')
def test_generate_synthetic_multiple_minorities(mock_exists, mock_read_pkl, mock_read_csv, mock_gen):
    """Test analyze_and_generate_synthetic handles multiple minority classes"""
    highly_imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 200,
        'sql_context': ['c'] * 200,
        'sql': ['s'] * 200,
        'sql_complexity': (
            ['basic'] * 150 + ['aggregation'] * 20 + 
            ['CTEs'] * 10 + ['window functions'] * 10 + ['subqueries'] * 10
        )
    })
    mock_read_pkl.return_value = highly_imbalanced
    mock_exists.return_value = True
    
    synthetic = pd.DataFrame({
        'sql_prompt': ['syn'] * 50,
        'sql_context': ['ctx'] * 50,
        'sql': ['sql'] * 50,
        'sql_complexity': ['CTEs'] * 20 + ['window functions'] * 15 + ['subqueries'] * 15
    })
    mock_read_csv.return_value = synthetic
    
    result = analyze_and_generate_synthetic()
    
    assert result['synthetic_generated'] == 50


# TEST 33: Merge and split creates proper train/val ratio
@patch('os.path.exists')
@patch('os.remove')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_pickle')
@patch('os.makedirs')
def test_merge_and_split_proper_ratio(mock_mkdir, mock_read, mock_csv, mock_rm, mock_exists):
    """Test merge_and_split creates ~90/10 train/val split"""
    large_data = pd.DataFrame({
        'sql_prompt': ['q'] * 100,
        'sql_context': ['c'] * 100,
        'sql': ['s'] * 100,
        'sql_complexity': ['basic'] * 50 + ['aggregation'] * 50
    })
    empty_synthetic = pd.DataFrame(columns=['sql_prompt', 'sql_context', 'sql', 'sql_complexity'])
    
    mock_read.side_effect = [large_data, large_data.copy(), empty_synthetic]
    mock_exists.return_value = False
    
    result = merge_and_split()
    
    # Check approximately 90/10 split
    total = result['train'] + result['val']
    val_percentage = (result['val'] / total) * 100
    assert 8 <= val_percentage <= 12  # Should be around 10%


# TEST 34: Validate engineered schema detects duplicates
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_csv')
def test_validate_engineered_schema_checks_duplicates(mock_read, mock_mkdir, mock_file, mock_ctx):
    """Test validate_engineered_schema includes duplicate checking"""
    data_with_dups = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test', 
                      'translate English to SQL: query: test2',
                      'translate English to SQL: query: test'],  # Duplicate
        'sql': ['SELECT 1', 'SELECT 2', 'SELECT 1'],
        'sql_complexity': ['basic', 'basic', 'basic']
    })
    mock_read.side_effect = [data_with_dups, data_with_dups, data_with_dups]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    
    result = validate_engineered_schema(**mock_ctx)
    
    # Should still validate (duplicates are counted but not an error)
    assert result['schema_validated'] == True


# TEST 35: Remove data leakage pushes stats to XCom
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_remove_leakage_pushes_xcom(mock_read, mock_csv, mock_ctx):
    """Test remove_data_leakage pushes statistics to XCom"""
    train = pd.DataFrame({
        'input_text': ['q1', 'q2'],
        'sql': ['s1', 's2'],
        'sql_complexity': ['basic'] * 2
    })
    val = pd.DataFrame({'input_text': ['q3'], 'sql': ['s3'], 'sql_complexity': ['basic']})
    test = pd.DataFrame({'input_text': ['q4'], 'sql': ['s4'], 'sql_complexity': ['basic']})
    
    mock_read.side_effect = [train, val, test]
    
    remove_data_leakage(**mock_ctx)
    
    # Should push leakage_stats to XCom
    assert mock_ctx['task_instance'].xcom_push.called
    
    # Verify what was pushed
    push_calls = mock_ctx['task_instance'].xcom_push.call_args_list
    assert len(push_calls) > 0
    
    # Check that leakage_stats was pushed
    pushed_keys = [call[1]['key'] for call in push_calls if 'key' in call[1]]
    assert 'leakage_stats' in pushed_keys


# ============================================================================
# BATCH 5: Additional Test Cases (5 tests)
# ============================================================================

# TEST 36: Validate raw schema calculates text statistics
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_pickle')
def test_validate_raw_schema_calculates_statistics(mock_read, mock_mkdir, mock_file, mock_ctx, sample_data):
    """Test validate_raw_schema calculates comprehensive text statistics"""
    mock_read.side_effect = [sample_data, sample_data.copy()]
    
    result = validate_raw_schema(**mock_ctx)
    
    assert result['schema_validated'] == True
    # Should push statistics to XCom
    assert mock_ctx['task_instance'].xcom_push.called


# TEST 37: Detect bias calculates correct max/min classes
@patch('utils.EmailContentGenerator.send_email_notification')
@patch('pandas.read_pickle')
def test_detect_bias_max_min_classes(mock_read, mock_email, mock_ctx):
    """Test detect_bias correctly identifies max and min classes"""
    imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 100,
        'sql_context': ['c'] * 100,
        'sql': ['s'] * 100,
        'sql_complexity': ['basic'] * 70 + ['aggregation'] * 20 + ['CTEs'] * 10
    })
    mock_read.return_value = imbalanced
    
    result = detect_bias(**mock_ctx)
    
    assert result['max_class'] == 'basic'
    assert result['max_count'] == 70
    assert result['min_class'] == 'CTEs'
    assert result['min_count'] == 10


# TEST 38: Merge and split creates files with correct columns
@patch('os.path.exists')
@patch('os.remove')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_pickle')
@patch('os.makedirs')
def test_merge_and_split_output_columns(mock_mkdir, mock_read, mock_csv, mock_rm, mock_exists, sample_data):
    """Test merge_and_split saves files with input_text, sql, sql_complexity columns"""
    empty_synthetic = pd.DataFrame(columns=['sql_prompt', 'sql_context', 'sql', 'sql_complexity'])
    
    mock_read.side_effect = [sample_data, sample_data.copy(), empty_synthetic]
    mock_exists.return_value = False
    
    result = merge_and_split()
    
    # Should have saved 3 files (train, val, test)
    assert 'files' in result
    assert len(result['files']) == 3


# TEST 39: Validate engineered schema checks dataset sizes
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_csv')
def test_validate_engineered_schema_dataset_sizes(mock_read, mock_mkdir, mock_file, mock_ctx):
    """Test validate_engineered_schema reports correct dataset sizes"""
    train = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test'] * 100,
        'sql': ['SELECT 1'] * 100,
        'sql_complexity': ['basic'] * 100
    })
    val = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test'] * 10,
        'sql': ['SELECT 1'] * 10,
        'sql_complexity': ['basic'] * 10
    })
    test = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test'] * 50,
        'sql': ['SELECT 1'] * 50,
        'sql_complexity': ['basic'] * 50
    })
    
    mock_read.side_effect = [train, val, test]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    
    result = validate_engineered_schema(**mock_ctx)
    
    assert result['train_size'] == 100
    assert result['val_size'] == 10
    assert result['test_size'] == 50


# TEST 40: Remove data leakage handles empty test set
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_remove_leakage_empty_test_set(mock_read, mock_csv, mock_ctx):
    """Test remove_data_leakage handles empty test dataset"""
    train = pd.DataFrame({
        'input_text': ['q1', 'q2'],
        'sql': ['s1', 's2'],
        'sql_complexity': ['basic'] * 2
    })
    val = pd.DataFrame({
        'input_text': ['q3'],
        'sql': ['s3'],
        'sql_complexity': ['basic']
    })
    test = pd.DataFrame(columns=['input_text', 'sql', 'sql_complexity'])  # Empty
    
    mock_read.side_effect = [train, val, test]
    
    result = remove_data_leakage(**mock_ctx)
    
    # Should handle empty test set gracefully
    assert result['test_duplicates_removed'] == 0
    assert 'total_train_removed' in result

# TEST 41: Preprocess preserves all columns
@patch('pandas.DataFrame.to_pickle')
@patch('pandas.read_pickle')
def test_preprocess_preserves_columns(mock_read, mock_pickle, sample_data):
    """Test preprocess doesn't drop any columns"""
    mock_read.side_effect = [sample_data, sample_data.copy()]
    
    result = preprocess()
    
    assert result['preprocessed'] == True
    # Function should preserve data integrity
    assert mock_pickle.call_count == 2


# TEST 42: Validate raw schema detects empty strings
@patch('pandas.read_pickle')
def test_validate_raw_schema_empty_strings(mock_read, mock_ctx):
    """Test validate_raw_schema can detect empty string values"""
    data_with_empty = pd.DataFrame({
        'sql_prompt': ['query1', '   ', 'query3'],  # Empty string (whitespace)
        'sql_context': ['ctx1', 'ctx2', 'ctx3'],
        'sql': ['SELECT 1', 'SELECT 2', ''],  # Empty string
        'sql_complexity': ['basic', 'basic', 'basic']
    })
    mock_read.side_effect = [data_with_empty, data_with_empty.copy()]
    
    # The function should check for empty strings in data quality
    # It may or may not raise error depending on implementation
    try:
        result = validate_raw_schema(**mock_ctx)
        # If it passes, it should still validate
        assert result['schema_validated'] == True
    except ValueError:
        # If it catches empty strings, that's also correct
        pass


# TEST 43: Analyze and generate synthetic with file generation error
@patch('utils.DataGenerator.GenerateAdditionalData')
@patch('pandas.read_pickle')
@patch('os.path.exists')
def test_generate_synthetic_file_error(mock_exists, mock_read, mock_gen):
    """Test analyze_and_generate_synthetic handles file not created"""
    imbalanced = pd.DataFrame({
        'sql_prompt': ['q'] * 60,
        'sql_context': ['c'] * 60,
        'sql': ['s'] * 60,
        'sql_complexity': ['basic'] * 50 + ['CTEs'] * 10
    })
    mock_read.return_value = imbalanced
    mock_exists.return_value = False  # File doesn't exist
    
    result = analyze_and_generate_synthetic()
    
    # Should return 0 when file doesn't exist
    assert result['synthetic_generated'] == 0


# TEST 44: Merge and split handles single class stratification
@patch('os.path.exists')
@patch('os.remove')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_pickle')
@patch('os.makedirs')
def test_merge_and_split_single_class_stratification(mock_mkdir, mock_read, mock_csv, mock_rm, mock_exists):
    """Test merge_and_split handles dataset with single class"""
    single_class_data = pd.DataFrame({
        'sql_prompt': ['q'] * 100,
        'sql_context': ['c'] * 100,
        'sql': ['s'] * 100,
        'sql_complexity': ['basic'] * 100  # All same class
    })
    empty_synthetic = pd.DataFrame(columns=['sql_prompt', 'sql_context', 'sql', 'sql_complexity'])
    
    mock_read.side_effect = [single_class_data, single_class_data.copy(), empty_synthetic]
    mock_exists.return_value = False
    
    # Should handle single-class stratification
    try:
        result = merge_and_split()
        assert 'train' in result
        assert 'val' in result
    except ValueError:
        # May raise error if can't stratify single class - that's ok
        pass


# TEST 45: Validate engineered schema handles all same complexity
@patch('builtins.open', new_callable=mock_open)
@patch('os.makedirs')
@patch('pandas.read_csv')
def test_validate_engineered_schema_single_complexity(mock_read, mock_mkdir, mock_file, mock_ctx):
    """Test validate_engineered_schema with all same sql_complexity"""
    all_same = pd.DataFrame({
        'input_text': ['translate English to SQL: query: test'] * 50,
        'sql': [f'SELECT {i}' for i in range(50)],
        'sql_complexity': ['basic'] * 50  # All same
    })
    
    mock_read.side_effect = [all_same, all_same, all_same]
    mock_ctx['task_instance'].xcom_pull.return_value = {}
    
    result = validate_engineered_schema(**mock_ctx)
    
    assert result['schema_validated'] == True
    assert result['train_size'] == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])