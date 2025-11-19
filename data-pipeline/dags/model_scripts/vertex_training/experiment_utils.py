from google.cloud import aiplatform

def start_experiment_run(experiment_name, run_name, project_id, region):
    """
    Initializes the experiment (creating it if needed) and starts a new run.
    """
    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name
    )
    run = aiplatform.start_run(run=run_name)
    print(f"Experiment run started: {experiment_name}/{run_name}")
    return run

def log_experiment_params(run, params_dict: dict):
    """
    Logs hyperparameters to the given ExperimentRun object.
    """
    print(f"Logging parameters: {params_dict}")
    run.log_params(params_dict)

def log_experiment_metrics(run, metrics_dict: dict):
    """
    Logs metrics to the given ExperimentRun object.
    """
    print(f"Logging metrics: {metrics_dict}")
    run.log_metrics(metrics_dict)

def get_experiment_run(run_name, experiment_name, project_id, region):
    """
    Reconstructs the same experiment run using its name.
    """
    # init() sets the project and location context
    aiplatform.init(project=project_id, location=region)
    
    # Get the run object directly by its name and experiment name
    run = aiplatform.ExperimentRun(
        run_name=run_name,
        experiment=experiment_name
    )
    return run