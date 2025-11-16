from datetime import datetime
import os
from google.cloud import aiplatform
from google.cloud import storage
from airflow.exceptions import AirflowException

def download_from_gcs_if_needed(path):
    """
    Download files from GCS if path starts with gs://
    Returns local path
    """
    if not path.startswith("gs://"):
        return path
    
    print(f"Downloading from GCS: {path}")
    
    # Parse GCS path
    path_parts = path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    # Create local directory
    local_dir = f"/tmp/{prefix.split('/')[-1]}"
    os.makedirs(local_dir, exist_ok=True)
    
    # Download all files from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    downloaded_count = 0
    for blob in blobs:
        if not blob.name.endswith("/"):  # Skip directory markers
            relative_path = blob.name[len(prefix):].lstrip("/")
            local_file_path = os.path.join(local_dir, relative_path)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            print(f"  Downloading: {blob.name} -> {local_file_path}")
            blob.download_to_filename(local_file_path)
            downloaded_count += 1
    
    print(f"✅ Downloaded {downloaded_count} files to {local_dir}")
    return local_dir

def submit_vertex_training_job(project_id, region, container_image_uri, gcs_model_dir, gcs_output_dir):
    """
    Submit Vertex AI custom training job (LoRA fine-tuning)
    """
    aiplatform.init(project=project_id, location=region)

    training_args = [
        "--train_data=gs://train_data_query_hub/data/train.csv",
        "--val_data=gs://train_data_query_hub/data/val.csv",
        f"--model_dir={gcs_model_dir}",
        f"--output_dir={gcs_output_dir}",
        "--num_train_epochs=1",
        "--per_device_train_batch_size=8",
        "--per_device_eval_batch_size=8",
        "--learning_rate=5e-5",
        "--lora_r=16",
        "--lora_alpha=32",
        "--lora_dropout=0.1",
        "--target_modules", "q", "v"
    ]

    job = aiplatform.CustomJob(
        display_name="hf_training_job",
        staging_bucket="gs://train_data_query_hub/staging_bucket",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "a2-highgpu-1g",
                "accelerator_type": "NVIDIA_TESLA_A100",
                "accelerator_count": 1
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_image_uri,
                "command": ["python3", "train.py"],
                "args": training_args
            }
        }]
    )

    print("Submitting Vertex AI Custom Job...")
    job.run(sync=True)
    print(f"✅ Training finished. Model saved to: {gcs_output_dir}")
    return gcs_output_dir