from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta

#dss

from model_scripts.retrain_model import (
    fetch_latest_model_and_image,
    train_on_vertex_ai,
    build_push_docker_image
)

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=6),
}

def create_model_training_dag():

    with DAG(
        'vertex_ai_model_training_pipeline',
        default_args=default_args,
        description='Retrain model on Vertex AI using latest image + GCS files',
        catchup=False,
        tags=['vertex_ai', 'model_training', 'text2sql'],
        max_active_runs=1,
    ) as dag:

        start_pipeline = EmptyOperator(task_id='start_pipeline')

        # Task 1: Fetch latest image + model files
        fetch_model_task = PythonOperator(
            task_id='fetch_latest_model',
            python_callable=fetch_latest_model_and_image,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region")
            }
        )

        # Task 2: Train model on Vertex AI
        train_model_task = PythonOperator(
            task_id='train_on_vertex_ai',
            python_callable=train_on_vertex_ai,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region")
            }
        )

        # train_model_task = PythonOperator(
        #     task_id='train_locally',
        #     python_callable=train_locally,
        #     op_kwargs={
        #         "project_id": Variable.get("gcp_project"),
        #         "region": Variable.get("gcp_region")
        #     }
        # )

        # Task 3: Build & push new Docker image
        push_image_task = PythonOperator(
            task_id='build_push_docker_image',
            python_callable=build_push_docker_image,
            op_kwargs={
                "project_id": Variable.get("gcp_project"),
                "region": Variable.get("gcp_region"),
                "artifact_registry_repo": Variable.get("artifact_registry_repo"),
                "build_bucket": Variable.get("cloud_build_bucket")
            }
        )

        training_completed = EmptyOperator(task_id='training_completed')
        training_failed = EmptyOperator(task_id='training_failed', trigger_rule='one_failed')

        # DAG dependencies
        start_pipeline >> fetch_model_task >> train_model_task >> push_image_task >> training_completed
        [fetch_model_task, train_model_task, push_image_task] >> training_failed

        return dag

model_training_dag = create_model_training_dag()