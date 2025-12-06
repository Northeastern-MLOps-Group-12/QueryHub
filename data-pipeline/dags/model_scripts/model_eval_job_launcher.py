from google.cloud import aiplatform
from airflow.models import Variable
from datetime import datetime

def build_output_csv_path(base_folder: str) -> str:
    """
    base_folder example: gs://bucket/output_data/
    Returns: gs://bucket/output_data/output-<timestamp>.csv
    """
    if base_folder.endswith("/"):
        base_folder = base_folder[:-1]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base_folder}/output-{timestamp}.csv"

def launch_evaluation_job(
    project_id, 
    region, 
    output_csv,
    model_registry_id, 
    run_name, 
    test_data_path,
    machine_type,
    gpu_type,
    gcs_staging_bucket,
    **kwargs
):
    """
    Submits the model evaluation as a Vertex AI Custom Job.
    This function is called by the Airflow PythonOperator.
    """
    aiplatform.init(project=project_id, location=region)

    # Get the container image URI.
    try:
        container_image_uri = Variable.get("vertex_ai_training_image_uri")
    except Exception as e:
        print(f"Could not find Airflow Variable 'vertex_ai_training_image_uri'. Error: {e}")
        raise

    output_csv = build_output_csv_path(output_csv)
    kwargs['ti'].xcom_push(key='evaluation_output_csv', value=output_csv)

    # Define the command-line arguments for the script
    # These will be passed to model_eval.py's main() function
    script_args = [
        f"--project_id={project_id}",
        f"--region={region}",
        f"--output_csv={output_csv}",
        f"--model_registry_id={model_registry_id}",
        f"--run_name={run_name}",
        f"--test_data_path={test_data_path}"
    ]

    # Configure the Vertex AI Custom Job
    eval_job = aiplatform.CustomJob(
        display_name=f"model-eval-{run_name}",
        staging_bucket=gcs_staging_bucket,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": gpu_type,
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_image_uri,
                    "command": ["python3", "model_eval.py"],
                    "args": script_args,
                },
            }
        ],
    )

    print(f"Submitting Vertex AI evaluation job for run: {run_name}...")
    eval_job.run(service_account=Variable.get("service_account"), sync=True)

    print(f"✅ Evaluation job completed successfully.")