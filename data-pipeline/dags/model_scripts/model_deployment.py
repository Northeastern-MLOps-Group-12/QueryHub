from google.cloud import aiplatform
from .dag_experiment_utils import (
    get_experiment_run,
    log_experiment_metrics,
    log_experiment_params
)

def ensure_vertex_endpoint(project_id: str, region: str, endpoint_display_name: str, run_name) -> str:
    """
    Ensures a Vertex AI endpoint with the given display name exists.
    Returns the endpoint resource name.
    """
    aiplatform.init(project=project_id, location=region)

    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')

    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

    # Log to Vertex AI Experiment
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)

    print(f"Endpoint Name = {endpoint.resource_name}")
    log_experiment_params(run, {"endpoint_name": endpoint.resource_name})

    # This will be stored as XCom return_value by PythonOperator
    return endpoint.resource_name


def deploy_model_to_vertex_endpoint(project_id: str, region: str, run_name, endpoint_name: str, model_resource_name: str, machine_type: str, min_replica_count: int = 1, 
                                    max_replica_count: int = 2, traffic_percentage: int = 100) -> str:
    """
    Deploys a model from the Vertex Model Registry to a given endpoint.
    If endpoint already has models, this will simply add a new deployed model
    and set the provided traffic percentage for it.
    """
    aiplatform.init(project=project_id, location=region)

    model = aiplatform.Model(model_resource_name)
    endpoint = aiplatform.Endpoint(endpoint_name)

    endpoint.deploy(model=model, machine_type=machine_type, min_replica_count=min_replica_count, max_replica_count=max_replica_count, traffic_percentage=traffic_percentage, sync=True)

    print(f"✅ Model {model.display_name} deployed to endpoint: {endpoint.display_name}")

        # Log to Vertex AI Experiment
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)

    log_experiment_params(run, {"model_display_name": model.display_name})

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

    # Return endpoint name just for convenience / logging
    return endpoint.resource_name