# Vertex AI Model Training Pipeline

## Overview

This Apache Airflow DAG orchestrates an end-to-end machine learning pipeline for training, evaluating, and validating a Text-to-SQL model on Google Cloud Vertex AI. The pipeline automates the complete workflow from fetching the latest model to bias detection and syntax validation.

---

## Pipeline Architecture

```
Start Pipeline
    ↓
Fetch Latest Model
    ↓
Train on Vertex AI (with LoRA fine-tuning)
    ↓
Upload Model to Vertex AI Registry
    ↓
Evaluate Model on Test Data
    ↓
Bias Detection (SQL Complexity Analysis)
    ↓
Syntax Validation
    ↓
Pipeline Completed
```

---

## Pipeline Tasks

### 1. **Fetch Latest Model** (`fetch_latest_model`)
- Retrieves the latest model file path from GCS
- Prepares the base model for fine-tuning
- **Output**: Model directory path

### 2. **Train on Vertex AI** (`train_on_vertex_ai`)
- Launches a Vertex AI Custom Training Job
- Performs LoRA (Low-Rank Adaptation) fine-tuning on T5-Large
- Uses GPU acceleration
- Logs hyperparameters to Vertex AI Experiments
- **Output**: 
  - Saves trained model path in xcom

### 3. **Upload Model to Vertex AI** (`upload_model_to_vertex_ai`)
- Registers the trained model in Vertex AI Model Registry
- **Output**: Registered model path

### 4. **Evaluate Model** (`evaluate_model_on_vertex_ai`)
- Runs model evaluation on a test dataset
- Computes metrics: Exact Match (EM) and Token F1 Score
- Generates evaluation results CSV with predictions
- **Output**:
  - Saves prediction results in GCS
  - Logs EM and F1 score to Vertex AI Experiments

### 5. **Bias Detection** (`bias_detection`)
- Analyzes model performance across SQL complexity buckets
- Detects performance disparities between simple and complex queries
- Generates two reports:
  - Per-bucket performance metrics
  - SQL complexity distribution
- **Output**:
  - Uploads artifacts to GCS
  - Logs GCS path to Vertex AI Experiments

### 6. **Syntax Validation** (`syntax_validation`)
- Validates syntactic correctness of generated SQL queries
- Uses SQLGlot parser for syntax checking
- Computes syntax validity rate
- **Output**:
  - Logs syntax score in Experiments
  - Uploads artifacts to GCS

---

## DAG Configuration

```python
default_args = {
    'owner': 'data-engineering',
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=6),
}
```

### Key Settings
- **Max Active Runs**: 1 (prevents concurrent training jobs)
- **Catchup**: Disabled
- **Execution Timeout**: 6 hours
- **Retries**: 2 attempts with 5-minute delay

---

## XCom Data Flow

The pipeline uses XCom to pass data between tasks:

| Source Task | XCom Key | Used By |
|-------------|----------|---------|
| `train_on_vertex_ai` | `trained_model_gcs` | `upload_model_to_vertex_ai` |
| `train_on_vertex_ai` | `experiment_run_name` | `evaluate_model`, `bias_detection`, `syntax_validation` |
| `upload_model_to_vertex_ai` | `registered_model_name` | `evaluate_model` |
| `evaluate_model_on_vertex_ai` | `evaluation_output_csv` | `bias_detection`, `syntax_validation` |

---

## Error Handling

### Failure Triggers
The `training_failed` task is triggered when any of these tasks fail:
- fetch_model_task
- train_model_task
- upload_model_task
- evaluate_model
- bias_detection_task
- syntax_validation

### Retry Logic
- Each task retries up to 2 times
- 5-minute delay between retries
- Individual task timeout: 6 hours

---

## Monitoring & Logging

### Vertex AI Experiments
All training runs are logged to Vertex AI Experiments:
- Training metrics and hyperparameters
- Evaluation metrics (EM, F1 scores)
- Bias detection and syntax validation results are pushed to GCS bucket

### Sample Experiment Tracking
1. **Metrics**

    ![Metrics](https://lucid.app/publicSegments/view/604b9877-d0cd-4a51-b8ad-111cf2978ec4/image.png)

1. **Parameters**

    ![Parameters](https://lucid.app/publicSegments/view/d986f29b-41d8-4b55-beb3-bd27b7a3522a/image.png)

---

## Output Artifacts

### Training Outputs
- **Model Checkpoints**: `gs://bucket/{run_name}/`
- **Training Logs**: Vertex AI console

### Evaluation Outputs
- **Predictions CSV**:
  - Columns: `input_text`, `expected_sql`, `sql_complexity`, `predicted_sql`, `exact_match`, `f1_score`

### Validation Outputs
- **Bias Detection**:
  - `per_bucket_complexity_eval.csv`: Performance by complexity
  - `sql_complexity_distribution.csv`: Dataset distribution
- **Syntax Validation**:
  - `syntax_validation_results.csv`: Syntax check results

---

## Customization

### Modify Training Parameters
Edit `model_scripts/retrain_model.py`:
```python
training_args = [
    "--num_train_epochs=3",            # Number of epochs
    "--per_device_train_batch_size=8", # Batch size
    "--learning_rate=5e-5",            # Learning rate
    "--lora_r=16",                     # LoRA rank
    # ... other hyperparameters
]
```

### Change Machine Type
Edit GPU configuration:
```python
"machine_spec": {
    "machine_type": "g2-standard-8",    # Larger machine
    "accelerator_type": "NVIDIA_L4",    # GPU
    "accelerator_count": 2              # Multiple GPUs
}
```