from airflow.exceptions import AirflowException
from google.cloud import storage
from airflow.models import Variable
from datetime import datetime
from google.cloud import aiplatform
from model_scripts.train_utils import submit_vertex_training_job
from model_scripts.dag_experiment_utils import (
    start_experiment_run, 
    log_experiment_params,
    get_experiment_run
)

def fetch_latest_model(project_id, gcs_bucket_name, region, **kwargs):
    """
    Fetch the latest Artifact Registry Docker image and latest merged model folder.
    Push them to XCom for downstream tasks.
    """
    ti = kwargs["ti"]
    client = storage.Client(project=project_id)
    bucket_name = gcs_bucket_name

    # Latest merged model folder
    merged_prefix = "registered_models/"
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=merged_prefix))

    model_folders = {}
    for blob in blobs:
        relative_path = blob.name[len(merged_prefix):].strip("/")
        if "/" not in relative_path:
            continue
        folder_name = relative_path.split("/")[0]
        # Keep only the latest by creation time
        if folder_name not in model_folders or blob.time_created > model_folders[folder_name]:
            model_folders[folder_name] = blob.time_created

    if not model_folders:
        raise AirflowException(f"No merged models found in gs://{bucket_name}/{merged_prefix}")

    latest_model_folder = max(model_folders, key=lambda k: model_folders[k])
    latest_model_path = f"gs://{bucket_name}/{merged_prefix}{latest_model_folder}"
    print(f"✅ Latest merged model path: {latest_model_path}")
    ti.xcom_push(key="latest_model_dir", value=latest_model_path)

    return latest_model_path

def get_latest_subfolder(gcs_path):
    """
    Returns the latest subfolder under a GCS prefix by inspecting blob names.
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError("Invalid GCS path")
    
    bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    folder_names = set()
    for blob in blobs:
        # blob.name e.g. processed_datasets/20251208_152334/train.csv
        relative = blob.name[len(prefix):]

        # only folders have a "/" in the relative path
        if "/" in relative:
            folder = relative.split("/")[0]
            folder_names.add(folder)

    if not folder_names:
        raise AirflowException(f"No subfolders found under {gcs_path}")

    latest_folder = sorted(folder_names)[-1]

    return f"gs://{bucket_name}/{prefix}{latest_folder}"

def train_on_vertex_ai(project_id, region, gcp_processed_data_path, container_image_uri, machine_type, gpu_type, gcs_staging_bucket, gcs_registered_models, train_samples, val_samples, num_train_epochs, **kwargs):
    """
    Submit Vertex AI Custom Training Job using latest model files + image.
    """
    ti = kwargs["ti"]
    gcs_model_dir = ti.xcom_pull(task_ids="fetch_latest_model", key="latest_model_dir")

    if not gcs_model_dir or not container_image_uri:
        raise AirflowException("Missing GCS model path or container image URI from XCom")
    
    latest_dataset_folder = get_latest_subfolder(gcp_processed_data_path)

    gcs_train_data = f"{latest_dataset_folder}/train.csv"
    gcs_val_data = f"{latest_dataset_folder}/val.csv"
    
    # Start experiment run
    print("Starting experiment run...")
    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run = start_experiment_run(
        experiment_name="queryhub-experiments",
        run_name=run_name,
        project_id=project_id,
        region=region
    )
    ti.xcom_push(key="experiment_run_name", value=run_name)

    # Prepare output model path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    gcs_output_dir = f"{gcs_registered_models}/{timestamp}"

    # Log params to experiment
    print("Logging training parameters to experiment...")
    params_to_log = {
        "container_image_uri": container_image_uri,
        "input_model_gcs": gcs_model_dir,
        "output_model_gcs": gcs_output_dir
    }
    log_experiment_params(run, params_to_log)

    print(f"Training model from: {gcs_model_dir} using image: {container_image_uri}")

    # Submit training job
    trained_model_path = submit_vertex_training_job(
        project_id=project_id,
        region=region,
        container_image_uri=container_image_uri,
        machine_type=machine_type,
        gpu_type=gpu_type,
        gcs_model_dir=gcs_model_dir,
        gcs_train_data=gcs_train_data,
        gcs_val_data=gcs_val_data,
        gcs_output_dir=gcs_output_dir,
        gcs_staging_bucket=gcs_staging_bucket,
        train_samples=train_samples,
        val_samples=val_samples,
        num_train_epochs=num_train_epochs,
        run_name=run_name
    )

    # Push output path for next task (image build)
    ti.xcom_push(key="trained_model_gcs", value=trained_model_path)

def register_model_in_vertex_ai(project_id, region, model_artifact_path, serving_container_image_uri, **kwargs):
    """
    Upload trained model artifacts to Vertex AI Model Registry.
    """
    ti = kwargs["ti"]

    aiplatform.init(project=project_id, location=region)

    print(f"Uploading model from {model_artifact_path} to Vertex AI Registry...")

    model = aiplatform.Model.upload(
        display_name=f"queryhub-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        artifact_uri=model_artifact_path,
        serving_container_image_uri=serving_container_image_uri,
    )
    print(f"✅ Model uploaded to Vertex AI Model Registry: {model.resource_name}")

    # Log version info to experiment using a dictionary
    run_name = ti.xcom_pull(task_ids="train_on_vertex_ai", key="experiment_run_name")
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)
    log_experiment_params(run, {"vertex_model_resource": model.resource_name})

    # return model.resource_name
    ti.xcom_push(key="registered_model_name", value=model.resource_name)
    return model.resource_name
