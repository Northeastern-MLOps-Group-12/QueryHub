from google.cloud import storage
import pandas as pd
import numpy as np
from sqlglot import parse_one
from google.cloud import aiplatform
from pathlib import Path
import sys
from model_scripts.bias_detection import upload_to_gcs
from model_scripts.dag_experiment_utils import (
    log_experiment_metrics,
    get_experiment_run
)

def run_syntax_validation_task(project_id, region, run_name, gcs_csv_path, gcs_output_path, **kwargs):
    """
    Airflow wrapper to run SQL syntax validation and log metrics to Vertex AI Experiment.
    """
    ti = kwargs["ti"]
    print(f"Running SQL syntax validation on: {gcs_csv_path}")
    
    syntax_overall, per_complexity_valid = syntax_validation_from_gcs(gcs_csv_path)

    print(f"Overall syntax validity: {syntax_overall:.4f}")
    print("Per-complexity syntax validity:")
    print(per_complexity_valid)

    # Log to Vertex AI Experiment
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)

    print(f"syntax_overall = {syntax_overall}")
    log_experiment_metrics(run, {"syntax_overall": syntax_overall})

    # Save per-complexity DF as CSV and log as artifact
    tmp_file = "/tmp/per_complexity_validation.csv"
    per_complexity_valid.to_csv(tmp_file)

    # Upload to GCS
    folder_name = ti.xcom_pull(key="bias_and_syntax_validation_folder", task_ids="bias_detection")
    gcs_output_path = gcs_output_path.rstrip("/")
    gcs_output_path = f"{gcs_output_path}/{folder_name}/per_complexity_validation.csv"
    upload_to_gcs(tmp_file, gcs_output_path)

    # Log to Vertex AI Experiment
    print(f"Per Complexity Validation evaluation path = {gcs_output_path}")
    log_experiment_metrics(run, {"per_complexity_validation": gcs_output_path})

    # # Re-initialize with the experiment name
    # print(f"Resuming and ending experiment run: {run_name}")
    # aiplatform.init(
    #     project=project_id, 
    #     location=region, 
    #     experiment="queryhub-experiments"
    # )
    
    # # Resume the run to make it active
    # run = aiplatform.start_run(run=run_name, resume=True)
    
    # # End the (now active) run
    # aiplatform.end_run()
    
    # print("✅ Experiment run ended.")
    print("✅ Syntax validation completed.")

def syntax_validation_from_gcs(gcs_path, dialect="mysql"):
    """
    gcs_path example: 'gs://my-bucket/eval/predictions.csv'

    Returns:
        syntax_overall (float)
        per_complexity_valid (pd.DataFrame)
    """
    # 1. Parse bucket + blob
    if not gcs_path.startswith("gs://"):
        raise ValueError("gcs_path must start with gs://")

    path_no_prefix = gcs_path.replace("gs://", "")
    bucket_name, blob_path = path_no_prefix.split("/", 1)

    # 2. Download CSV from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    local_tmp_file = "/tmp/sql_eval.csv"
    blob.download_to_filename(local_tmp_file)

    # 3. Load CSV with predictions
    df = pd.read_csv(local_tmp_file)

    if "predicted_sql" not in df.columns:
        raise ValueError("CSV must contain a 'predicted_sql' column")

    if "sql_complexity" not in df.columns:
        raise ValueError("CSV must contain a 'sql_complexity' column")

    predictions = df["predicted_sql"].tolist()

    # 4. SQL syntax validator
    def is_executable(sql_str, dialect=dialect):
        try:
            parse_one(sql_str, dialect=dialect)
            return 1.0
        except Exception:
            return 0.0

    # Run validation
    syntax_scores = [is_executable(p) for p in predictions]

    # 5. Global syntax validity
    syntax_overall = float(np.mean(syntax_scores))

    # Add to DF
    df["valid_syntax"] = syntax_scores

    # 6. Per-complexity validity
    per_complexity_valid = (
        df.groupby("sql_complexity")["valid_syntax"]
          .agg(["mean", "count"])
          .round(4)
          .sort_values("mean", ascending=False)
    )

    return syntax_overall, per_complexity_valid


def get_model_id_from_latest_endpoint(project_id: str, region: str) -> str:
    """Get the model ID from the most recently created endpoint in Vertex AI."""
    
    aiplatform.init(project=project_id, location=region)
    
    # List all endpoints, sorted by create_time descending
    endpoints = aiplatform.Endpoint.list(
        order_by="create_time desc",
    )
    
    if not endpoints:
        raise ValueError("No endpoints found in the project")
    
    latest_endpoint = endpoints[0]
    print(f"Latest endpoint: {latest_endpoint.display_name}")
    print(f"Endpoint resource name: {latest_endpoint.resource_name}")
    
    # Get deployed models on this endpoint
    deployed_models = latest_endpoint.gca_resource.deployed_models
    
    if not deployed_models:
        raise ValueError(f"No models deployed to endpoint {latest_endpoint.display_name}")
    
    # Get the model ID from the first deployed model
    model_resource_name = deployed_models[0].model
    model_id = model_resource_name.split("/")[-1]
    
    print(f"Model resource name: {model_resource_name}")
    print(f"Model ID: {model_id}")
    
    return model_id

def get_metrics_from_experiment_by_model_id(project_id: str, region: str, model_id: str) -> dict:
    """
    Fetch metrics from experiment runs that match the given model_id.
    Looks for model_id within the vertex_model_resource parameter.
    """
    
    aiplatform.init(project=project_id, location=region)
    
    # List all experiments
    experiments = aiplatform.Experiment.list()
    
    if not experiments:
        raise ValueError("No experiments found in the project")
    
    print(f"Found {len(experiments)} experiments")
    
    for experiment in experiments:
        print(f"\nChecking experiment: {experiment.name}")
        
        # Get all runs for this experiment
        runs = aiplatform.ExperimentRun.list(experiment=experiment.name)
        
        for run in runs:
            # Get run parameters
            params = run.get_params()
            
            # Check if vertex_model_resource contains our model_id
            vertex_model_resource = params.get("vertex_model_resource", "")
            
            if model_id in vertex_model_resource:
                print(f"\nFound matching run: {run.name}")
                print(f"vertex_model_resource: {vertex_model_resource}")
                
                # Get metrics
                metrics = run.get_metrics()
                
                exact_match = metrics.get("exact_match")
                f1_score = metrics.get("f1_score")
                
                print(f"exact_match: {exact_match}")
                print(f"f1_score: {f1_score}")
                
                return {
                    "experiment_name": experiment.name,
                    "run_name": run.name,
                    "exact_match": exact_match,
                    "f1_score": f1_score,
                    "all_metrics": metrics,
                    "all_params": params
                }
    
    raise ValueError(f"No experiment run found with model_id: {model_id}")



def get_metrics_from_experiment_run(
    project_id: str, 
    region: str, 
    run_name: str,
) -> dict:
    """
    Fetch metrics from a specific experiment run by run name.
    """
    aiplatform.init(project=project_id, location=region)
    
    # Get the experiment run directly by name
    run = aiplatform.ExperimentRun(experiment="queryhub-experiments", run_name=run_name)
    
    metrics = run.get_metrics()
    
    return {
        "exact_match": metrics.get("exact_match"),
        "f1_score": metrics.get("f1_score"),
    }


def choose_best_model(project_id, region, run_name, **kwargs):
    """
    Compare new model metrics with currently deployed model metrics.
    Returns task_id to branch to.
    """
    ti = kwargs["ti"]
    
    aiplatform.init(project=project_id, location=region)
    
    # Get metrics from the NEW model (from the current training run)
    print(f"Fetching metrics for new model from run: {run_name}")
    new_metrics = get_metrics_from_experiment_run(project_id, region, run_name)
    new_exact_match = new_metrics.get("exact_match", 0)
    new_f1_score = new_metrics.get("f1_score", 0)
    
    print(f"New Model Metrics:")
    print(f"  Exact Match: {new_exact_match}")
    print(f"  F1 Score: {new_f1_score}")
    
    # Get metrics from the OLD model (currently deployed)
    try:
        model_id = get_model_id_from_latest_endpoint(project_id, region)
        old_metrics = get_metrics_from_experiment_by_model_id(project_id, region, model_id)
        old_exact_match = old_metrics.get("exact_match", 0)
        old_f1_score = old_metrics.get("f1_score", 0)
        
        print(f"\nOld Model Metrics (Model ID: {model_id}):")
        print(f"  Exact Match: {old_exact_match}")
        print(f"  F1 Score: {old_f1_score}")
        
    except ValueError as e:
        # No existing endpoint/model - new model wins by default
        print(f"No existing deployed model found: {e}")
        print("New model will be deployed as the first model.")
        return "ensure_vertex_endpoint"
    
    # Compare metrics - new model is better if F1 score is higher and exact_match not worse
    is_new_model_better = (new_f1_score > old_f1_score) and (new_exact_match >= old_exact_match)
    
    print(f"\n--- Comparison ---")
    print(f"F1 Score: {new_f1_score} (new) vs {old_f1_score} (old)")
    print(f"Exact Match: {new_exact_match} (new) vs {old_exact_match} (old)")
    print(f"New model is better: {is_new_model_better}")
    
    if is_new_model_better:
        print("✅ New model is better - proceeding to deployment")
        return "deploy_model_to_endpoint"
    else:
        print("⏭️ Old model is better - skipping deployment")
        return "skip_deployment"