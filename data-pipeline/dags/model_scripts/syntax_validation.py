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
    
    print("✅ Experiment run ended.")
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