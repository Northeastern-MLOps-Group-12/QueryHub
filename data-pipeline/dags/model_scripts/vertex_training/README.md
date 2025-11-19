# Text-to-SQL Fine-Tuning with LoRA on Vertex AI

This project fine-tunes a T5-Large model for text-to-SQL conversion using LoRA (Low-Rank Adaptation) technique on Google Cloud Vertex AI. The training script is designed for containerized deployment, enabling scalable GPU-accelerated training with automatic artifact management.

---

## Overview

The training pipeline:
1. Downloads model artifacts and datasets from Google Cloud Storage (GCS)
2. Performs LoRA fine-tuning on T5-Large
3. Merges LoRA adapters into the base model
4. Uploads the final merged model back to GCS

**Key Features:**
- ✅ Containerized training for Vertex AI Custom Jobs
- ✅ Automatic GCS download/upload
- ✅ Memory-efficient LoRA fine-tuning
- ✅ GPU acceleration (NVIDIA)
- ✅ Merged model output (no separate adapter loading required)

---

## Prerequisites

### Required Services
- Google Cloud Project with billing enabled
- Vertex AI API enabled
- Cloud Storage buckets created
- Service account with appropriate permissions

### Required Permissions
```
- aiplatform.customJobs.create
- storage.objects.get
- storage.objects.create
- storage.buckets.get
```

---

## Dependencies

The training script requires:

```txt
torch
transformers
datasets
peft
pandas
google-cloud-storage
scikit-learn
```

These are typically installed in the training container image.

---

## Data Format

### Input CSV Files

Training data should be CSV files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `input_text` | Natural language query | "Show all employees in sales department" |
| `sql` | Corresponding SQL query | "SELECT * FROM employees WHERE department = 'sales'" |
| `sql_complexity` (optional) | Complexity label for bias detection | "basic SQL", "aggregation", "joins" |

---

## Sample Sizes

Due to resource constraints, this uses sampled datasets:
- **Training**: ~40,000 samples
- **Validation**: ~20,000 samples
- **Test**: ~5,700 samples

Adjust these in the sampling section based on your available resources.

---

## Configuration

### Command-Line Arguments

The training script accepts the following arguments:

#### Required Arguments

```bash
--model_dir          # GCS path to base model (gs://bucket/path/to/model/)
--train_data         # GCS path to training CSV (gs://bucket/data/train.csv)
--val_data           # GCS path to validation CSV (gs://bucket/data/val.csv)
--output_dir         # GCS path for output model (gs://bucket/outputs/model/)
```

#### Model Configuration

```bash
--max_source_length  # Max tokens for input text (default: 512)
--max_target_length  # Max tokens for SQL output (default: 256)
```

#### Training Hyperparameters

```bash
--num_train_epochs              # Number of training epochs (default: 3)
--per_device_train_batch_size   # Training batch size per GPU (default: 8)
--per_device_eval_batch_size    # Evaluation batch size per GPU (default: 8)
--learning_rate                 # Learning rate (default: 5e-5)
--warmup_steps                  # Number of warmup steps (default: 500)
--weight_decay                  # Weight decay for regularization (default: 0.01)
```

#### LoRA Configuration

```bash
--lora_r             # LoRA rank (default: 16)
--lora_alpha         # LoRA alpha scaling (default: 32)
--lora_dropout       # LoRA dropout rate (default: 0.1)
--target_modules     # Target attention modules (default: q v)
```

**Example Usage:**
```bash
python train.py \
  --model_dir=gs://my-bucket/models/t5-large \
  --train_data=gs://my-bucket/data/train.csv \
  --val_data=gs://my-bucket/data/val.csv \
  --output_dir=gs://my-bucket/outputs/finetuned-model \
  --num_train_epochs=3 \
  --per_device_train_batch_size=8 \
  --learning_rate=5e-5 \
  --lora_r=16
```

---

## Training Process Explained

### 1. GCS Download Phase

```python
def download_from_gcs_if_needed(path):
    """Downloads files or folders from GCS to local /tmp directory"""
```

**What happens:**
- Detects if path starts with `gs://`
- Downloads single files (CSV, JSON, TXT) to `/tmp/filename`
- Downloads entire folders (model directories) to `/tmp/foldername`
- Returns local path for subsequent processing

**Example:**
```
gs://bucket/data/train.csv → /tmp/train.csv
gs://bucket/models/t5/ → /tmp/t5/
```

### 2. Model and Tokenizer Loading

```python
model = AutoModelForSeq2SeqLM.from_pretrained(
    local_model_dir,
    torch_dtype=torch.bfloat16,   # Half precision
    low_cpu_mem_usage=True,       # Memory optimization
    local_files_only=True,        # Use downloaded files
    trust_remote_code=True
)
```

**Optimizations applied:**
- ✅ Mixed precision (bfloat16) for 50% memory reduction
- ✅ Gradient checkpointing enabled
- ✅ Cache disabled for training
- ✅ Right-side truncation for inputs

### 3. Data Preprocessing

The `preprocess_batch()` function handles tokenization:

```python
def preprocess_batch(batch):
    # Tokenize inputs (natural language)
    model_inputs = tokenizer(
        batch["input_text"], 
        max_length=max_source_length, 
        truncation=True, 
        padding="max_length"
    )
    
    # Tokenize targets (SQL queries)
    labels = tokenizer(
        batch["sql"], 
        max_length=max_target_length, 
        truncation=True, 
        padding="max_length"
    )
    
    # Replace padding tokens with -100 (ignored in loss)
    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) 
         for token in seq] 
        for seq in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels_ids
    return model_inputs
```

**Key Points:**
- Input text padded/truncated to 512 tokens
- SQL queries padded/truncated to 256 tokens
- Padding tokens masked with -100 to exclude from loss calculation

### 4. LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,                               # Rank of low-rank matrices
    lora_alpha=32,                      # Scaling factor (α/r)
    lora_dropout=0.1,                   # Dropout for regularization
    bias="none",                        # Don't train bias terms
    task_type=TaskType.SEQ_2_SEQ_LM,    # Seq2Seq task
    target_modules=["q", "v"]           # Apply to Q and V projections
)
```

**Why LoRA?**
- **Memory Efficient**: Trains only ~1-2% of model parameters
- **Fast Training**: Significantly reduces training time
- **Minimal Performance Loss**: Achieves comparable results to full fine-tuning
- **Portable**: LoRA adapters are small (10-50 MB vs. 3 GB for full model)

**How LoRA Works:**
LoRA injects trainable low-rank decomposition matrices into the attention mechanism:
```
W_new = W_frozen + (B × A) × (α/r)
```
Where:
- `W_frozen` = Original pre-trained weights (frozen)
- `B × A` = Low-rank matrices (trainable, rank r)
- `α/r` = Scaling factor

### 5. Training Configuration

```python
training_args = TrainingArguments(
    output_dir="/tmp/lora_trained",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    fp16=True,                         # Mixed precision training
    dataloader_num_workers=4,          # Parallel data loading
)
```

**Training Strategies:**
- **Mixed Precision (FP16)**: Reduces memory by ~50% while maintaining stability
- **Gradient Checkpointing**: Trades computation for memory
- **Periodic Logging**: Track loss every 100 steps
- **Checkpointing**: Save model every 1000 steps

### 6. LoRA Adapter Merging

**Critical Step:** Unlike the Colab version, this script merges the LoRA adapter into the base model:

```python
# After training completes
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("/tmp/merged_model")
tokenizer.save_pretrained("/tmp/merged_model")
```

**Why Merge?**
- ✅ **Simplified Inference**: No need to load adapter separately
- ✅ **Production Ready**: Works with standard transformers inference
- ✅ **Single Artifact**: One model directory instead of base + adapter
- ✅ **Better Performance**: Eliminates adapter overhead at inference time

**Before (Separate Adapter):**
```python
base_model = AutoModelForSeq2SeqLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "lora-adapter")
```

**After (Merged):**
```python
model = AutoModelForSeq2SeqLM.from_pretrained("merged-model")
```

### 7. GCS Upload Phase

```python
def upload_to_gcs(local_path, gcs_path):
    """Uploads entire directory to GCS, preserving structure"""
```

**What happens:**
- Walks through local directory tree
- Uploads each file to GCS
- Preserves folder structure
- Final model available at `gs://bucket/outputs/model/`

---

## Vertex AI Deployment

### Launching Training Job

**Python Script (using Vertex AI SDK):**
```python
from google.cloud import aiplatform

aiplatform.init(project="your-project", location="us-central1")

training_args = [
    "--train_data=gs://bucket/data/train.csv",
    "--val_data=gs://bucket/data/val.csv",
    "--model_dir=gs://bucket/models/t5-large",
    "--output_dir=gs://bucket/outputs/finetuned-model",
    "--num_train_epochs=3",
    "--per_device_train_batch_size=8",
    "--learning_rate=5e-5",
    "--lora_r=16",
    "--lora_alpha=32",
    "--target_modules", "q", "v"
]

job = aiplatform.CustomJob(
    display_name="text2sql-lora-training",
    staging_bucket="gs://bucket/staging",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "g2-standard-4",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/your-project/text2sql-training:latest",
            "command": ["python3", "train.py"],
            "args": training_args
        }
    }]
)

job.run()
```

---

## Model Evaluation

After training completes, the model is evaluated on a held-out test dataset to measure its performance on unseen SQL generation tasks.

### Evaluation Pipeline

The evaluation process consists of:

1. **Model Loading from Registry**
2. **Test Data Loading from GCS**
3. **SQL Generation and Metrics Computation**
4. **Results Upload to GCS**
5. **Metrics Logging to Vertex AI Experiments**

### Evaluation Metrics

Two primary metrics are computed:

#### 1. Exact Match (EM)

**Definition:** Binary metric indicating perfect prediction match.

```python
def exact_match(pred, gold):
    """Returns 1.0 if prediction exactly matches expected SQL"""
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
```

**Characteristics:**
- **Strict**: Requires character-perfect match (case-insensitive)
- **Range**: 0.0 or 1.0 per sample
- **Aggregate**: Average across all test samples
- **Use Case**: Measures production-ready SQL queries

**Example:**
```
Expected:  "SELECT * FROM employees WHERE salary > 50000"
Predicted: "SELECT * FROM employees WHERE salary > 50000"
EM Score: 1.0 ✅

Expected:  "SELECT * FROM employees WHERE salary > 50000"
Predicted: "SELECT * FROM employees WHERE salary >= 50000"
EM Score: 0.0 ❌ (even though semantically similar)
```

#### 2. Token-level F1 Score

**Definition:** Measures partial correctness by treating SQL as a bag-of-tokens.

```python
def compute_f1(pred, gold):
    """
    Tokenizes SQL queries and computes F1 score
    Treats SQL as unordered set of tokens
    """
    pred_tokens = sql_tokenize(pred)
    gold_tokens = sql_tokenize(gold)
    
    # Convert to binary vectors
    all_tokens = set(pred_tokens + gold_tokens)
    pred_vec = [1 if t in pred_tokens else 0 for t in all_tokens]
    gold_vec = [1 if t in gold_tokens else 0 for t in all_tokens]
    
    return f1_score(gold_vec, pred_vec, average="micro")
```

**SQL Tokenization:**
```python
def sql_tokenize(s):
    """Splits SQL into tokens, separating special characters"""
    s = s.lower().strip()
    for tok in ["(", ")", ",", ";"]:
        s = s.replace(tok, f" {tok} ")
    return s.split()
```

**Characteristics:**
- **Lenient**: Rewards partial correctness
- **Range**: 0.0 to 1.0 (continuous)
- **Order-Agnostic**: Token order doesn't matter
- **Use Case**: Measures semantic similarity

**Example:**
```
Expected:  "SELECT name, salary FROM employees WHERE salary > 50000"
Predicted: "SELECT salary, name FROM employees WHERE salary > 50000"

Tokenization:
Expected:  [select, name, ,, salary, from, employees, where, salary, >, 50000]
Predicted: [select, salary, ,, name, from, employees, where, salary, >, 50000]

Common tokens: 11/11 (all tokens present, just reordered)
F1 Score: 1.0 ✅ (captures semantic equivalence)
```

### Evaluation Process Details

#### 1. Load Model from Vertex AI Registry

```python
def load_model_from_registry(model_resource_name, project_id, region, device="cpu"):
    """
    Downloads trained model artifacts from Vertex AI Model Registry
    Returns: tokenizer, model (ready for inference)
    """
```

**What happens:**
- Retrieves model URI from Vertex AI Model Registry
- Downloads all model files from GCS to `/tmp/eval_model/`
- Loads tokenizer and model into memory
- Moves model to appropriate device (CPU/GPU)

#### 2. Load Test Dataset

```python
def load_test_dataset(gcs_path):
    """Downloads test CSV from GCS and loads into pandas DataFrame"""
```

#### 3. Run Evaluation Loop

```python
def evaluate_model(tokenizer, model, df, output_csv, device="cpu"):
    """
    For each test sample:
    1. Tokenize input text
    2. Generate SQL prediction
    3. Compute EM and F1 scores
    4. Save results to CSV
    """
```

**Per-sample evaluation:**
```python
# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate SQL
with torch.no_grad():
    outputs = model.generate(**inputs)
pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compute metrics
em = exact_match(pred, expected)
f1 = compute_f1(pred, expected)
```

#### Step 4: Generate Results CSV

Output CSV contains detailed results for analysis:

| Column | Description |
|--------|-------------|
| `input_text` | Natural language query |
| `expected_sql` | Ground truth SQL |
| `predicted_sql` | Model-generated SQL |
| `sql_complexity` | Complexity category |
| `exact_match` | EM score (0.0 or 1.0) |
| `f1_score` | Token F1 score (0.0-1.0) |

#### Step 5: Upload Results and Log Metrics

```python
# Save locally
results_df.to_csv("/tmp/evaluation_results.csv", index=False)

# Upload to GCS
upload_to_gcs("/tmp/evaluation_results.csv", "gs://bucket/eval/results.csv")

# Log aggregate metrics to Vertex AI Experiments
run = get_experiment_run(run_name, experiment_name="queryhub-experiments")
log_experiment_metrics(run, {
    "exact_match": final_em,
    "f1_score": final_f1
})
```

### Running Evaluation

#### Via Airflow DAG

Evaluation is triggered automatically in the training pipeline:

```python
evaluate_model = PythonOperator(
    task_id="evaluate_model_on_vertex_ai",
    python_callable=launch_evaluation_job,
    op_kwargs={
        "project_id": Variable.get("gcp_project"),
        "region": Variable.get("gcp_region"),
        "output_csv": Variable.get("gcp_evaluation_output_csv"),
        "model_registry_id": "{{ ti.xcom_pull(task_ids='upload_model_to_vertex_ai', key='registered_model_name') }}",
        "run_name": "{{ ti.xcom_pull(task_ids='train_on_vertex_ai', key='experiment_run_name') }}",
        "test_data_path": Variable.get("gcp_test_data_path"),
        "machine_type": Variable.get("vertex_ai_eval_machine_type"),
        "gpu_type": Variable.get("vertex_ai_eval_gpu_type"),
    }
)
```

---

## Technical Details

### Model Architecture
- **Base**: T5-Large (770M parameters)
- **Trainable**: ~4-8M parameters (LoRA adapters)
- **Target Modules**: Query (Q) and Value (V) projection matrices in self-attention
- **Output**: Merged model with 770M parameters (base weights + LoRA deltas)

### Optimization
- **Optimizer**: AdamW (default)
- **Learning Rate Schedule**: Linear warmup + linear decay
- **Weight Decay**: 0.01 (default)
- **Gradient Clipping**: 1.0 (default)

---

## Acknowledgments

- **Base Model**: [gaussalgo/T5-LM-Large-text2sql-spider](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spider)
- **LoRA Implementation**: [Hugging Face PEFT](https://github.com/huggingface/peft)
- **Framework**: [Hugging Face Transformers](https://github.com/huggingface/transformers)
- **Training Platform**: [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)