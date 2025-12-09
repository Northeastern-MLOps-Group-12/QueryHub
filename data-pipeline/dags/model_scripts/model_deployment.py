from google.cloud import aiplatform
from .dag_experiment_utils import (
    get_experiment_run,
    log_experiment_metrics,
    log_experiment_params
)

def deploy_model_to_endpoint(project_id, region, run_name, model_resource_name, machine_type, **kwargs):
    """
    Deploy the newly registered model to the Vertex AI Endpoint.
    """
    aiplatform.init(project=project_id, location=region)

    print(f"🚀 Deploying model to endpoint")
    print(f"   Model: {model_resource_name}")
    print(f"   Machine Type: {machine_type}")
    
    # Load model from registry
    model = aiplatform.Model(model_resource_name)
    print(f"📦 Model loaded: {model.display_name}")

    endpoint = model.deploy(
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1,
    )

    # Log to Vertex AI Experiment
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)

    log_experiment_params(run, {"deployed_endpoint": endpoint.resource_name})

    # Re-initialize with the experiment name
    print(f"Resuming and ending experiment run: {run_name}")
    aiplatform.init(
        project=project_id, 
        location=region, 
        experiment="queryhub-experiments"
    )
    
    # Resume the run to make it active
    run = aiplatform.start_run(run=run_name, resume=True)
    
    # End the (now active) run
    aiplatform.end_run()
    
    print(f"✅ Deployment initiated!")
    print(f"Endpoint resource name: {endpoint.resource_name}")
    print(f"Endpoint ID: {endpoint.name}")