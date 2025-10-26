from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from utils.EmailContentGenerator import send_email_notification , notify_task_failure
from utils.SQLValidator import _validate_single_sql
import logging

# ============================================================================
# DAG DEFAULT ARGS WITH FAILURE NOTIFICATION
# ============================================================================

default_args = {
    'owner': 'queryhub',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': notify_task_failure,  # Send email on any task failure
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
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import cpu_count
    
    train_df = pd.read_pickle('/tmp/train_raw.pkl')
    test_df = pd.read_pickle('/tmp/test_raw.pkl')
    
    # Track anomalies (invalid SQL queries)
    train_anomalies = []
    test_anomalies = []
    
    # Determine number of workers (use 75% of available CPUs)
    max_workers = max(1, int(cpu_count() * 0.75))
    logging.info(f"Using {max_workers} parallel workers for SQL validation")
    
    # Prepare data for parallel processing
    train_tasks = [(idx, sql, 'train') for idx, sql in enumerate(train_df['sql'])]
    test_tasks = [(idx, sql, 'test') for idx, sql in enumerate(test_df['sql'])]
    
    # Validate train data in parallel
    logging.info(f"Validating {len(train_tasks)} training SQL queries in parallel...")
    train_valid_mask = [False] * len(train_df)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_validate_single_sql, task): task for task in train_tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                idx, is_valid, error_msg, error_type = future.result()
                train_valid_mask[idx] = is_valid
                
                if not is_valid:
                    train_anomalies.append({
                        'source': 'train',
                        'index': idx,
                        'sql_query': task[1],
                        'error_message': error_msg,
                        'error_type': error_type
                    })
            except Exception as e:
                logging.error(f"Error processing train task {task[0]}: {e}")
                train_valid_mask[task[0]] = False
    
    train_valid_mask = pd.Series(train_valid_mask)
    
    # Validate test data in parallel
    logging.info(f"Validating {len(test_tasks)} test SQL queries in parallel...")
    test_valid_mask = [False] * len(test_df)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_validate_single_sql, task): task for task in test_tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                idx, is_valid, error_msg, error_type = future.result()
                test_valid_mask[idx] = is_valid
                
                if not is_valid:
                    test_anomalies.append({
                        'source': 'test',
                        'index': idx,
                        'sql_query': task[1],
                        'error_message': error_msg,
                        'error_type': error_type
                    })
            except Exception as e:
                logging.error(f"Error processing test task {task[0]}: {e}")
                test_valid_mask[task[0]] = False
    
    test_valid_mask = pd.Series(test_valid_mask)
    
    logging.info(f"Train: {train_valid_mask.sum()}/{len(train_df)} valid queries")
    logging.info(f"Test: {test_valid_mask.sum()}/{len(test_df)} valid queries")
    logging.info(f"Train anomalies detected: {len(train_anomalies)}")
    logging.info(f"Test anomalies detected: {len(test_anomalies)}")
    
    # Save anomalies report to CSV
    output_dir = '/opt/airflow/data'
    os.makedirs(output_dir, exist_ok=True)
    
    all_anomalies = train_anomalies + test_anomalies
    if all_anomalies:
        anomalies_df = pd.DataFrame(all_anomalies)
        anomalies_path = f'{output_dir}/sql_validation_anomalies.csv'
        anomalies_df.to_csv(anomalies_path, index=False)
        logging.info(f"✅ Saved {len(all_anomalies)} anomalies to {anomalies_path}")
    else:
        logging.info("No anomalies detected - all SQL queries are valid")
    
    # Filter valid data
    train_df = train_df[train_valid_mask]
    test_df = test_df[test_valid_mask]
    
    train_df.to_pickle('/tmp/train_valid.pkl')
    test_df.to_pickle('/tmp/test_valid.pkl')
    
    return {
        'train_valid': len(train_df), 
        'test_valid': len(test_df),
        'train_anomalies': len(train_anomalies),
        'test_anomalies': len(test_anomalies),
        'total_anomalies': len(all_anomalies)
    }


def preprocess():
    import pandas as pd
    
    train_df = pd.read_pickle('/tmp/train_valid.pkl')
    test_df = pd.read_pickle('/tmp/test_valid.pkl')
    
    train_df.to_pickle('/tmp/train_preprocessed.pkl')
    test_df.to_pickle('/tmp/test_preprocessed.pkl')
    
    logging.info(f"Preprocessed {len(train_df)} train, {len(test_df)} test samples")
    return {'preprocessed': True}


def detect_bias(**context):
    """Detect class imbalance bias in the dataset and send email alert"""
    import pandas as pd
    from utils.EmailContentGenerator import generate_bias_detection_email, send_email_notification
    
    # Load preprocessed training data
    train_df = pd.read_pickle('/tmp/train_preprocessed.pkl')
    
    # Analyze sql_complexity distribution
    complexity_counts = train_df['sql_complexity'].value_counts()
    total_samples = len(train_df)
    
    logging.info(f"Analyzing bias in sql_complexity distribution...")
    logging.info(f"Total training samples: {total_samples}")
    logging.info(f"Complexity distribution:\n{complexity_counts}")
    
    # Calculate statistics
    max_count = complexity_counts.max()
    min_count = complexity_counts.min()
    max_class = complexity_counts.idxmax()
    min_class = complexity_counts.idxmin()
    
    # Calculate imbalance ratio
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Calculate percentage distribution
    percentages = (complexity_counts / total_samples * 100).round(2)
    
    # Determine bias level
    bias_detected = False
    bias_level = "None"
    minority_classes = []
    
    # Define bias thresholds
    SEVERE_BIAS_RATIO = 10
    MODERATE_BIAS_RATIO = 5
    MINORITY_THRESHOLD = 0.5
    
    if imbalance_ratio >= SEVERE_BIAS_RATIO:
        bias_detected = True
        bias_level = "SEVERE"
    elif imbalance_ratio >= MODERATE_BIAS_RATIO:
        bias_detected = True
        bias_level = "MODERATE"
    elif imbalance_ratio >= 2:
        bias_detected = True
        bias_level = "MILD"
    
    # Identify minority classes
    for complexity, count in complexity_counts.items():
        if count < (max_count * MINORITY_THRESHOLD):
            minority_classes.append(f"{complexity} ({count} samples, {percentages[complexity]}%)")
    
    # Store bias info
    bias_info = {
        'bias_detected': bias_detected,
        'bias_level': bias_level,
        'imbalance_ratio': float(imbalance_ratio),
        'minority_classes': len(minority_classes),
        'max_class': max_class,
        'max_count': int(max_count),
        'min_class': min_class,
        'min_count': int(min_count)
    }
    
    # Generate and send email using utility function
    subject, html_content = generate_bias_detection_email(
        complexity_counts, total_samples, bias_info, minority_classes
    )
    send_email_notification(subject, html_content)
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='bias_info', value=bias_info)
    
    logging.info(f"Bias detection complete: {bias_level} bias detected")
    return bias_info


def analyze_and_generate_synthetic():
    """Analyze class distribution and generate synthetic data"""
    import pandas as pd
    import sys
    import os
    sys.path.append('/opt/airflow/dags')
    from utils.DataGenerator import GenerateAdditionalData
    
    # Load preprocessed training data
    train_df = pd.read_pickle('/tmp/train_preprocessed.pkl')
    
    # Analyze sql_complexity distribution
    complexity_counts = train_df['sql_complexity'].value_counts()
    logging.info(f"Original complexity distribution:\n{complexity_counts}")
    
    # Find maximum count (target for balancing)
    max_count = complexity_counts.max()
    logging.info(f"Target count per complexity: {max_count}")
    
    # Calculate how many synthetic samples needed per complexity
    target_counts = {}
    minority_classes = ['CTEs', 'set operations', 'window functions', 
                       'subqueries', 'multiple_joins']
    
    for complexity in minority_classes:
        current_count = complexity_counts.get(complexity, 0)
        needed = max(0, max_count - current_count)
        if needed > 0:
            target_counts[complexity] = needed
            logging.info(f"{complexity}: need {needed} synthetic samples")
    
    if not target_counts:
        logging.info("Dataset already balanced, no synthetic data needed")
        synthetic_df = pd.DataFrame()
        synthetic_df.to_pickle('/tmp/synthetic_data.pkl')
        return {'synthetic_generated': 0}
    
    # Generate synthetic data
    logging.info(f"Generating synthetic data with target_counts: {target_counts}")
    GenerateAdditionalData(target_counts)
    
    # Load the generated CSV
    path = '/opt/airflow/data/synthetic_data.csv'
    
    if not os.path.exists(path):
        logging.error(f"Synthetic data file not found at {path}")
        synthetic_df = pd.DataFrame()
        synthetic_df.to_pickle('/tmp/synthetic_data.pkl')
        return {'synthetic_generated': 0}
    
    synthetic_df = pd.read_csv(path)
    logging.info(f"Loaded {len(synthetic_df)} synthetic samples from {path}")
    
    # Save as pickle for next task
    synthetic_df.to_pickle('/tmp/synthetic_data.pkl')
    
    logging.info(f"Generated {len(synthetic_df)} synthetic samples")
    return {'synthetic_generated': len(synthetic_df)}


def merge_and_split():
    """Merge original + synthetic for train, keep test separate"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os
    
    # Load data
    train_df = pd.read_pickle('/tmp/train_preprocessed.pkl')
    test_df = pd.read_pickle('/tmp/test_preprocessed.pkl')
    synthetic_df = pd.read_pickle('/tmp/synthetic_data.pkl')
    
    logging.info(f"Original train size: {len(train_df)}")
    logging.info(f"Synthetic data size: {len(synthetic_df)}")
    logging.info(f"Test size: {len(test_df)}")
    
    # Store original complexity distribution
    original_complexity_dist = train_df['sql_complexity'].value_counts().to_dict()
    
    # Merge original + synthetic for training
    if len(synthetic_df) > 0:
        train_cols = ['sql_prompt', 'sql_context', 'sql', 'sql_complexity']
        
        train_original = train_df[train_cols].copy()
        synthetic_data = synthetic_df[train_cols].copy()
        
        train_combined = pd.concat([train_original, synthetic_data], ignore_index=True)
        logging.info(f"Combined train size: {len(train_combined)}")
        
        final_dist = train_combined['sql_complexity'].value_counts()
        logging.info(f"Final complexity distribution:\n{final_dist}")
    else:
        train_combined = train_df[['sql_prompt', 'sql_context', 'sql', 'sql_complexity']].copy()
    
    # Store final complexity distribution
    final_complexity_dist = train_combined['sql_complexity'].value_counts().to_dict()
    
    # Create input_text column AFTER merging
    def format_input_text(row):
        prompt = "translate English to SQL: "
        if pd.notna(row['sql_context']):
            prompt += f"context: {row['sql_context']}\n\n"
        prompt += f"query: {row['sql_prompt']}"
        return prompt
    
    train_combined['input_text'] = train_combined.apply(format_input_text, axis=1)
    logging.info("Created input_text column from sql_context + sql_prompt")
    
    # Also create input_text for test data
    test_df['input_text'] = test_df.apply(format_input_text, axis=1)
    
    # Stratified split for train/val
    train_combined['strat'] = train_combined['sql_complexity'].astype(str)
    counts = train_combined['strat'].value_counts()
    valid_strat = counts[counts >= 2].index
    train_combined = train_combined[train_combined['strat'].isin(valid_strat)]
    
    train_final, val_final = train_test_split(
        train_combined, test_size=0.1, stratify=train_combined['strat'], random_state=42
    )
    
    # Drop stratification column
    train_final = train_final.drop(columns=['strat'])
    val_final = val_final.drop(columns=['strat'])
    
    # Save to /opt/airflow/data/ as CSV
    output_dir = '/opt/airflow/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old files
    for filename in ['train.csv', 'val.csv', 'test.csv']:
        old_file = f'{output_dir}/{filename}'
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                logging.info(f"Removed old file: {old_file}")
            except Exception as e:
                logging.warning(f"Could not remove {old_file}: {e}")
    
    # Save ONLY input_text and sql columns
    save_cols = ['input_text', 'sql']
    
    saved_files = []
    for name, df in [('train', train_final), ('val', val_final), ('test', test_df)]:
        path = f'{output_dir}/{name}.csv'
        df[save_cols].to_csv(path, index=False)
        saved_files.append(path)
        logging.info(f"Saved {len(df)} rows to {path}")
    
    return {
        'files': saved_files,
        'train': len(train_final),
        'val': len(val_final),
        'test': len(test_df),
        'train_original': len(train_df),
        'synthetic_added': len(synthetic_df),
        'original_complexity_dist': original_complexity_dist,
        'final_complexity_dist': final_complexity_dist
    }


def send_pipeline_success_notification(**context):
    """Send success notification email with pipeline statistics"""
    import pandas as pd
    from utils.EmailContentGenerator import send_email_notification
    
    # Get results from previous tasks via XCom
    task_instance = context['task_instance']
    
    # Get stats from different tasks
    load_stats = task_instance.xcom_pull(task_ids='load_data') or {}
    validate_stats = task_instance.xcom_pull(task_ids='validate_sql') or {}
    bias_info = task_instance.xcom_pull(task_ids='detect_bias', key='bias_info') or {}
    synthetic_stats = task_instance.xcom_pull(task_ids='analyze_and_generate_synthetic') or {}
    merge_stats = task_instance.xcom_pull(task_ids='merge_and_split') or {}
    
    # Calculate improvement metrics
    original_dist = merge_stats.get('original_complexity_dist', {})
    final_dist = merge_stats.get('final_complexity_dist', {})
    
    # Build success email
    subject = "✅ Pipeline Success - Data Processing Complete with Bias Mitigation"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #27ae60; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .stats-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #27ae60; }}
            .warning-box {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
            .success {{ color: #27ae60; font-weight: bold; }}
            .info {{ color: #0984e3; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #27ae60; color: white; }}
            .improved {{ background-color: #d4edda; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>✅ QueryHub Data Pipeline - Success Report</h2>
            <p>Text-to-SQL Dataset Processing Complete</p>
        </div>
        
        <div class="content">
            <h3 class="success">🎉 Pipeline Execution Successful!</h3>
            <p>Your data pipeline has completed successfully with bias mitigation applied.</p>
            
            <div class="stats-box">
                <h4>📊 Pipeline Summary</h4>
                <table>
                    <tr>
                        <th>Stage</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td><strong>Data Loading</strong></td>
                        <td>Initial Training Samples</td>
                        <td>{load_stats.get('train', 0):,}</td>
                    </tr>
                    <tr>
                        <td></td>
                        <td>Initial Test Samples</td>
                        <td>{load_stats.get('test', 0):,}</td>
                    </tr>
                    <tr>
                        <td><strong>SQL Validation</strong></td>
                        <td>Valid Training Queries</td>
                        <td>{validate_stats.get('train_valid', 0):,}</td>
                    </tr>
                    <tr>
                        <td></td>
                        <td>Valid Test Queries</td>
                        <td>{validate_stats.get('test_valid', 0):,}</td>
                    </tr>
                    <tr class="warning-box">
                        <td></td>
                        <td>⚠️ Anomalies Detected</td>
                        <td>{validate_stats.get('total_anomalies', 0):,}</td>
                    </tr>
                    <tr>
                        <td><strong>Bias Detection</strong></td>
                        <td>Bias Level</td>
                        <td>{bias_info.get('bias_level', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td></td>
                        <td>Imbalance Ratio</td>
                        <td>{bias_info.get('imbalance_ratio', 0):.2f}x</td>
                    </tr>
                    <tr class="improved">
                        <td><strong>Synthetic Generation</strong></td>
                        <td>Synthetic Samples Added</td>
                        <td class="metric">{synthetic_stats.get('synthetic_generated', 0):,}</td>
                    </tr>
                    <tr class="improved">
                        <td><strong>Final Dataset</strong></td>
                        <td>Training Samples</td>
                        <td class="metric">{merge_stats.get('train', 0):,}</td>
                    </tr>
                    <tr>
                        <td></td>
                        <td>Validation Samples</td>
                        <td>{merge_stats.get('val', 0):,}</td>
                    </tr>
                    <tr>
                        <td></td>
                        <td>Test Samples</td>
                        <td>{merge_stats.get('test', 0):,}</td>
                    </tr>
                </table>
            </div>
    """
    
    # Add anomalies section if any were found
    if validate_stats.get('total_anomalies', 0) > 0:
        html_content += f"""
            <div class="warning-box">
                <h4>⚠️ SQL Validation Anomalies</h4>
                <p>Detected <strong>{validate_stats.get('total_anomalies', 0)}</strong> invalid SQL queries that were excluded from the dataset:</p>
                <ul>
                    <li>Training set anomalies: {validate_stats.get('train_anomalies', 0)}</li>
                    <li>Test set anomalies: {validate_stats.get('test_anomalies', 0)}</li>
                </ul>
                <p>📄 Full anomaly report saved to: <code>/opt/airflow/data/sql_validation_anomalies.csv</code></p>
            </div>
        """
    
    # Add class distribution comparison
    html_content += """
            <div class="stats-box">
                <h4>📈 Class Distribution - Before vs After Bias Mitigation</h4>
                <table>
                    <tr>
                        <th>SQL Complexity</th>
                        <th>Original Count</th>
                        <th>Final Count</th>
                        <th>Change</th>
                    </tr>
    """
    
    all_classes = set(list(original_dist.keys()) + list(final_dist.keys()))
    for complexity in sorted(all_classes):
        orig_count = original_dist.get(complexity, 0)
        final_count = final_dist.get(complexity, 0)
        change = final_count - orig_count
        change_str = f"+{change}" if change > 0 else str(change)
        row_class = "improved" if change > 0 else ""
        
        html_content += f"""
                    <tr class="{row_class}">
                        <td>{complexity}</td>
                        <td>{orig_count:,}</td>
                        <td>{final_count:,}</td>
                        <td>{change_str}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="stats-box">
                <h4>📁 Generated Files</h4>
                <ul>
                    <li><code>/opt/airflow/data/train.csv</code></li>
                    <li><code>/opt/airflow/data/val.csv</code></li>
                    <li><code>/opt/airflow/data/test.csv</code></li>
                    <li><code>/opt/airflow/data/synthetic_data.csv</code></li>
    """
    
    if validate_stats.get('total_anomalies', 0) > 0:
        html_content += """
                    <li><code>/opt/airflow/data/sql_validation_anomalies.csv</code></li>
        """
    
    html_content += """
                </ul>
            </div>
            
            <div class="stats-box">
                <h4>✨ Key Achievements</h4>
                <ul>
                    <li>✅ Successfully loaded and validated SQL dataset</li>
    """
    
    if validate_stats.get('total_anomalies', 0) > 0:
        html_content += f"""
                    <li>✅ Identified and documented {validate_stats.get('total_anomalies', 0)} SQL anomalies</li>
        """
    
    if synthetic_stats.get('synthetic_generated', 0) > 0:
        html_content += f"""
                    <li>✅ Generated {synthetic_stats.get('synthetic_generated', 0):,} synthetic samples for bias mitigation</li>
                    <li>✅ Balanced class distribution from {bias_info.get('imbalance_ratio', 0):.2f}x to improved ratio</li>
        """
    
    exec_date = str(context.get('execution_date', 'N/A'))
    train_count = merge_stats.get('train', 0)
    val_count = merge_stats.get('val', 0)
    
    html_content += f"""
                    <li>✅ Created stratified train/validation split ({train_count:,} train, {val_count:,} val)</li>
                    <li>✅ All datasets ready for model training</li>
                </ul>
            </div>
            
            <p style="margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                <em>Pipeline completed successfully on {exec_date}</em><br>
                <em>This is an automated success notification from the QueryHub Data Pipeline</em>
            </p>
        </div>
    </body>
    </html>
    """
    
    # Send success notification
    send_email_notification(subject, html_content)
    
    logging.info("✅ Pipeline success notification sent")
    return {'notification_sent': True}


# ============================================================================
# DAG DEFINITION
# ============================================================================

dag = DAG(
    'data_pipeline_with_synthetic_v3',
    default_args=default_args,
    description='Text-to-SQL Pipeline with Synthetic Data Augmentation',
    schedule=None,
    catchup=False,
    tags=['data-pipeline', 'synthetic-generation', 'bias-detection'],
)

t1 = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

t2 = PythonOperator(
    task_id='validate_sql',
    python_callable=validate_sql,
    dag=dag
)

t3 = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess,
    dag=dag
)

t4 = PythonOperator(
    task_id='detect_bias',
    python_callable=detect_bias,
    dag=dag
)

t5 = PythonOperator(
    task_id='analyze_and_generate_synthetic',
    python_callable=analyze_and_generate_synthetic,
    dag=dag
)

t6 = PythonOperator(
    task_id='merge_and_split',
    python_callable=merge_and_split,
    dag=dag
)

t7 = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_pipeline_success_notification,
    dag=dag
)

# Pipeline flow
t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7