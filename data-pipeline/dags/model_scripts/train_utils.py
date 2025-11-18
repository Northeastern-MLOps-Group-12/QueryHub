from google.cloud import aiplatform
from model_scripts.vertex_training.experiment_utils import ( 
    log_experiment_params,
    get_experiment_run
)

def submit_vertex_training_job(project_id, region, container_image_uri, machine_type, gpu_type, gcs_model_dir, gcs_train_data, gcs_val_data, gcs_output_dir, gcs_staging_bucket, run_name):
    """
    Submit Vertex AI custom training job (LoRA fine-tuning)
    """

    aiplatform.init(project=project_id, location=region)

    training_args = [
        f"--train_data={gcs_train_data}",
        f"--val_data={gcs_val_data}",
        f"--model_dir={gcs_model_dir}",
        f"--output_dir={gcs_output_dir}",
        "--num_train_epochs=1",
        "--per_device_train_batch_size=32",
        "--per_device_eval_batch_size=16",
        "--gradient_accumulation_steps=4",
        "--learning_rate=25e-4",
        "--lora_r=8",
        "--lora_alpha=16",
        "--lora_dropout=0.0683",
        "--target_modules", "q", "v"
    ]

    # Log hyper parameters info to experiment
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)
    params_dict = {}
    for arg in training_args:
        if "=" in arg:
            key, value = arg.lstrip("-").split("=", 1)
            params_dict[key] = value
        else:
            params_dict[arg] = True

    log_experiment_params(run, params_dict)

    # Configure the Custom Job
    job = aiplatform.CustomJob(
        display_name="hf_training_job",
        staging_bucket={gcs_staging_bucket},
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": gpu_type,
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