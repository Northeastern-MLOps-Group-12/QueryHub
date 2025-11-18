import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .experiment_utils import log_experiment_metrics, get_experiment_run
from google.cloud import aiplatform
from google.cloud import storage
from sklearn.metrics import f1_score

def load_model_from_registry(model_resource_name, project_id, region, device="cpu"):
    """
    Download model artifacts from Vertex AI Model Registry to local storage.
    """
    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model(model_resource_name)
    gcs_path = model.uri

    local_dir = "/tmp/eval_model"
    os.makedirs(local_dir, exist_ok=True)

    # Client to interact with GCS
    client = storage.Client(project=project_id)
    
    # Parse GCS path
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    if not prefix.endswith('/'):
        prefix += '/'

    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    print(f"Downloading model files from {gcs_path} to {local_dir}...")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        
        # Get relative path to maintain structure
        relative_path = blob.name[len(prefix):]
        if not relative_path:
            continue
            
        local_file_path = os.path.join(local_dir, relative_path)
        
        # Create sub-directories if they exist in GCS
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        print(f"  Downloading {blob.name} to {local_file_path}")
        blob.download_to_filename(local_file_path)
    
    print("✅ Download complete.")

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model_local = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
    model_local.to(device)
    print(f"Model loaded and moved to {device}")

    return tokenizer, model_local


def load_test_dataset(gcs_path):
    """
    Load test dataset from a given GCS path of format gs://bucket/path/to/file.csv
    """
    client = storage.Client() 
    
    # Parse GCS path
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    object_name = path_parts[1]
    
    local_file = "/tmp/test.csv"
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    
    print(f"Downloading {gcs_path} to {local_file}...")
    blob.download_to_filename(local_file)
    print("✅ Download complete.")
    
    return pd.read_csv(local_file)

def upload_to_gcs(local_path: str, gcs_path: str):
    """
    Upload a local file to a given GCS path of format gs://bucket/path/to/file.
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError("gcs_path must start with gs://")

    print(f"🔼 Uploading to GCS: {gcs_path}")

    # Extract bucket + object name
    path_no_prefix = gcs_path.replace("gs://", "")
    bucket_name, object_name = path_no_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    blob.upload_from_filename(local_path)

    print(f"✅ Uploaded to gs://{bucket_name}/{object_name}")

def evaluate_model(tokenizer, model, df, output_csv, device="cpu"):
    """
    Evaluate the model on the test dataset and save results to CSV.
    """
    em_scores = []
    f1_scores = []
    total = len(df)

    print(f"Starting evaluation on {total} samples...\n")

    # Create a copy of the test dataframe with only the needed columns
    results_df = pd.DataFrame({
        "input_text": df["input_text"],
        "expected_sql": df["sql"],
        "sql_complexity": df["sql_complexity"],
        "predicted_sql": [""] * total,
        "exact_match": [0.0] * total,
        "f1_score": [0.0] * total,
    })

    for idx, row in df.iterrows():
        input_text = row["input_text"]
        expected = row["sql"]

        print(f"Sample {idx+1}/{total}")
        print(f"Input: {input_text}")
        print(f"Expected: {expected}")

        try:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            # Move tensors to the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs)

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            pred = ""

        em = exact_match(pred, expected)
        f1 = compute_f1(pred, expected)
        
        em_scores.append(em)
        f1_scores.append(f1)

        print(f"Predicted: {pred}")
        print(f"Exact Match Score: {em}")
        print(f"F1 Score: {f1}")

        results_df.at[idx, "predicted_sql"] = pred
        results_df.at[idx, "exact_match"] = em
        results_df.at[idx, "f1_score"] = f1

    # Generate timestamped CSV filename
    filename = os.path.basename(output_csv)
    local_path = f"/tmp/{filename}"
    results_df.to_csv(local_path, index=False)
    print(f"✅ Local evaluation results saved to {local_path}")

    # Upload to GCS
    if output_csv.startswith("gs://"):
        upload_to_gcs(local_path, output_csv)

    # Final aggregate scores
    final_em = sum(em_scores) / len(em_scores)
    final_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Final Exact Match Score: {final_em:.4f}")
    print(f"Final F1 Score: {final_f1:.4f}")

    return final_em, final_f1


def run_evaluation(model_registry_id, test_data_path, project_id, region, output_csv, run_name):
    """
    Main function to run the evaluation process.
    """
    # Detect if GPU is available in the Vertex AI job
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Downloading model from registry...")
    tokenizer, model = load_model_from_registry(model_registry_id, project_id, region, device)

    print("Loading test dataset...")
    df = load_test_dataset(test_data_path)

    print("Running evaluation...")
    final_em, final_f1 = evaluate_model(tokenizer, model, df, output_csv, device)

    run = get_experiment_run(run_name, experiment_name="queryhub-experiments", project_id=project_id, region=region)

    print(f"Exact Match score = {final_em}")
    log_experiment_metrics(run, {"exact_match": final_em})

    print(f"F1 score = {final_f1}")
    log_experiment_metrics(run, {"f1_score": final_f1})

    return final_em, final_f1


def exact_match(pred, gold):
    """
    Compute exact match score between prediction and gold standard.
    """
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def sql_tokenize(s):
    """
    Tokenize SQL query for F1 computation.
    """
    s = s.lower().strip()
    for tok in ["(", ")", ",", ";"]:
        s = s.replace(tok, f" {tok} ")
    return s.split()

def compute_f1(pred, gold):
    """
    Compute F1 score between prediction and gold standard SQL queries.
    """
    pred_tokens = sql_tokenize(pred)
    gold_tokens = sql_tokenize(gold)
    all_tokens = list(set(pred_tokens + gold_tokens))
    pred_vec = [1 if t in pred_tokens else 0 for t in all_tokens]
    gold_vec = [1 if t in gold_tokens else 0 for t in all_tokens]
    if sum(gold_vec) == 0:
        return 1.0
    return f1_score(gold_vec, pred_vec, average="micro")

if __name__ == "__main__":
    """
    Entry point for running evaluation from command line.
    """
    parser = argparse.ArgumentParser(description="Run Vertex AI Model Evaluation")
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_registry_id", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    
    args = parser.parse_args()
    
    print("Starting Vertex AI Evaluation Job...")
    run_evaluation(
        model_registry_id=args.model_registry_id,
        test_data_path=args.test_data_path,
        project_id=args.project_id,
        region=args.region,
        output_csv=args.output_csv,
        run_name=args.run_name
    )
    print("Evaluation Job Finished.")