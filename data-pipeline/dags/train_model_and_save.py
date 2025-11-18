import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta

from model_scripts.retrain_model import (
    fetch_latest_model,
    train_on_vertex_ai,
    register_model_in_vertex_ai,
)
from model_scripts.bias_detection import run_bias_detection
from model_scripts.model_eval_job_launcher import launch_evaluation_job
from model_scripts.syntax_validation import run_syntax_validation_task
from dags.utils.test_utils import run_unit_tests
from utils.EmailContentGenerator import notify_task_failure

# Get alert email
ALERT_EMAIL = os.getenv('ALERT_EMAIL', Variable.get("alert_email"))

def failure_callback(context):
    """Wrapper to call notify_task_failure with the email"""
    return notify_task_failure(context, to_emails=[ALERT_EMAIL])

# DAG default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=6),
    'on_failure_callback': failure_callback,
}

def create_model_training_dag():
    """
    DAG to retrain model on Vertex AI using latest Docker image and GCS data.
    """

    with DAG(
        'vertex_ai_model_training_pipeline',
        default_args=default_args,
        description='Retrain model on Vertex AI using latest image + GCS files',
        catchup=False,
        tags=['vertex_ai', 'model_training', 'text2sql'],
        max_active_runs=1,
    ) as dag:

        # Start of the pipeline
        start_pipeline = EmptyOperator(task_id='start_pipeline')

        # Pre-test node
        run_model_unit_tests = PythonOperator(
            task_id='run_model_unit_tests',
            python_callable=run_unit_tests,
        )

        # Task 1: Fetch latest image + model files
        fetch_model_task = PythonOperator(
            task_id='fetch_latest_model',
            python_callable=fetch_latest_model,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "gcs_bucket_name": Variable.get("gcs_bucket_name"),
            }
        )

        # Task 2: Train model on Vertex AI
        train_model_task = PythonOperator(
            task_id='train_on_vertex_ai',
            python_callable=train_on_vertex_ai,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "gcs_train_data": Variable.get("gcp_train_data_path"),
                "gcs_val_data": Variable.get("gcp_val_data_path"),
                "container_image_uri": Variable.get("vertex_ai_training_image_uri"),
                "machine_type": Variable.get("vertex_ai_train_machine_type"),
                "gpu_type": Variable.get("vertex_ai_train_gpu_type"),
                "gcs_staging_bucket": Variable.get("gcs_staging_bucket"),
                "gcs_registered_models": Variable.get("gcs_registered_models"),
            }
        )

        # Task 3: Upload model to Vertex AI Model Registry
        upload_model_task = PythonOperator(
            task_id='upload_model_to_vertex_ai',
            python_callable=register_model_in_vertex_ai,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "model_artifact_path": "{{ ti.xcom_pull(task_ids='train_on_vertex_ai', key='trained_model_gcs') }}",
                "serving_container_image_uri": Variable.get("serving_container_image_uri"),
            }
        )

        # Task 4: Evaluation
        evaluate_model = PythonOperator(
            task_id="evaluate_model_on_vertex_ai",
            python_callable=launch_evaluation_job,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "output_csv": Variable.get("gcp_evaluation_output_csv"),
                "model_registry_id": "{{ ti.xcom_pull(task_ids='upload_model_to_vertex_ai', key='registered_model_name') }}",
                "run_name": "{{ ti.xcom_pull(task_ids='train_on_vertex_ai', key='experiment_run_name') }}",
                "test_data_path": Variable.get("gcp_test_data_path"),
                "machine_type": Variable.get("vertex_ai_eval_machine_type"),
                "gpu_type": Variable.get("vertex_ai_eval_gpu_type"),
                "gcs_staging_bucket": Variable.get("gcs_staging_bucket"),
            }
        )

        # Task 5: Bias Detection
        bias_detection_task = PythonOperator(
            task_id="bias_detection",
            python_callable=run_bias_detection,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "run_name": "{{ ti.xcom_pull(task_ids='train_on_vertex_ai', key='experiment_run_name') }}",
                "gcs_csv_path": "{{ ti.xcom_pull(task_ids='evaluate_model_on_vertex_ai', key='evaluation_output_csv')}}",
                "gcs_output_path": Variable.get("gcs_bias_and_syntax_validation_output"),
            }
        )

        # Task 6: Syntax Validation
        syntax_validation = PythonOperator(
            task_id="syntax_validation",
            python_callable=run_syntax_validation_task,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "run_name": "{{ ti.xcom_pull(task_ids='train_on_vertex_ai', key='experiment_run_name') }}",
                "gcs_csv_path": "{{ ti.xcom_pull(task_ids='evaluate_model_on_vertex_ai', key='evaluation_output_csv')}}",
                "gcs_output_path": Variable.get("gcs_bias_and_syntax_validation_output"),
            }
        )

        # Training completion and failure handlers
        training_completed = EmptyOperator(task_id='training_completed', trigger_rule='all_success')
        training_failed = EmptyOperator(task_id='training_failed', trigger_rule='one_failed')

        # DAG flow
        start_pipeline >> run_model_unit_tests >> fetch_model_task >> train_model_task >> upload_model_task >> evaluate_model >> bias_detection_task >> syntax_validation >> training_completed
        
        # Send any failure to training_failed
        [run_model_unit_tests, fetch_model_task, train_model_task, upload_model_task, evaluate_model, bias_detection_task, syntax_validation] >> training_failed

        return dag

model_training_dag = create_model_training_dag()