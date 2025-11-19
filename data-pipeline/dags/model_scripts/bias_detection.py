from google.cloud import storage
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import sys
current_file = Path(__file__).resolve()

# Two parent directories
parent_dir = current_file.parent.parent
grandparent_dir = current_file.parent.parent.parent

# Add to sys.path
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(grandparent_dir))

from model_scripts.vertex_training.experiment_utils import (
    get_experiment_run,
    log_experiment_metrics,
)

from model_scripts.vertex_training.model_eval import (
    upload_to_gcs,
)

def run_bias_detection(project_id, region, run_name, gcs_csv_path, gcs_output_path, **kwargs):
    """
    Airflow wrapper to run bias detection after evaluation completes.
    """
    ti = kwargs["ti"]
    print(f"Running bias detection on: {gcs_csv_path}")

    per_bucket, complex_dist = detect_bias(gcs_csv_path)

    print("\n===== SQL Complexity Distribution =====")
    print(complex_dist)

    print("\n===== Performance by Complexity Bucket =====")
    print(per_bucket)

    # Save DataFrames locally as CSV
    tmp_dir = "/tmp"
    per_bucket_file = os.path.join(tmp_dir, "per_bucket_complexity_eval.csv")
    complex_dist_file = os.path.join(tmp_dir, "sql_complexity_distribution.csv")

    per_bucket.to_csv(per_bucket_file)
    complex_dist.to_csv(complex_dist_file)

    # Upload CSVs to GCS
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_name = f"bias-{ts}"
    ti.xcom_push(key="bias_and_syntax_validation_folder", value=folder_name)

    # Ensure gcs_output_path has no trailing slash
    gcs_output_path = gcs_output_path.rstrip("/")

    # Build final full GCS paths for each file
    per_bucket_gcs = f"{gcs_output_path}/{folder_name}/per_bucket_complexity_eval.csv"
    complex_dist_gcs = f"{gcs_output_path}/{folder_name}/sql_complexity_distribution.csv"

    print("Uploading results")
    print("Per Bucket Complexity eval to: ", per_bucket_gcs)
    print("SQL Complexity Distribution eval to: ", complex_dist_gcs)

    # Upload the files
    upload_to_gcs(per_bucket_file, per_bucket_gcs)
    upload_to_gcs(complex_dist_file, complex_dist_gcs)

    # Log to Vertex AI Experiment
    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)

    print(f"Per Bucket Complexity evaluation path = {per_bucket_gcs}")
    log_experiment_metrics(run, {"per_bucket_complexity_eval": per_bucket_gcs})

    print(f"SQL Complexity distribution path = {complex_dist_gcs}")
    log_experiment_metrics(run, {"sql_complexity_distribution": complex_dist_gcs})

    return {
        "per_bucket": per_bucket.to_json(),
        "complex_dist": complex_dist.to_json(),
    }

def detect_bias(gcs_path):
    """
    Analyze evaluation results CSV from GCS for bias detection.
    Computes performance metrics across different SQL complexity buckets.
    Returns two DataFrames: per-bucket metrics and complexity distribution.
    """
    # 1. Parse bucket + blob from gs:// URL
    if not gcs_path.startswith("gs://"):
        raise ValueError("gcs_path must start with gs://")

    path_no_prefix = gcs_path.replace("gs://", "")
    bucket_name, blob_path = path_no_prefix.split("/", 1)

    # 2. Download CSV from GCS into a temp local file
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    local_tmp_file = "/tmp/eval_results.csv"
    blob.download_to_filename(local_tmp_file)
    print(f"Downloaded file to {local_tmp_file}")

    # 3. Load CSV into pandas
    df_eval = pd.read_csv(local_tmp_file)

    # 4. Compute complexity distribution
    complex_dist = (
        df_eval["sql_complexity"]
          .value_counts()
          .to_frame("count")
          .assign(perc=lambda x: (x["count"] / x["count"].sum() * 100).round(2))
    )

    # 5. Compute per-bucket EM and token F1
    per_bucket = (
        df_eval
        .groupby("sql_complexity")[["exact_match", "f1_score"]]
        .agg(["mean", "count"])
        .sort_values(("f1_score", "mean"), ascending=False)
    )

    return per_bucket, complex_dist