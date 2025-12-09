import subprocess
import sys
import argparse
import os
import gc
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
from google.cloud import storage

HANDLER_CODE = r"""
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class Text2SQLHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = properties.get("model_dir")

        logger.info(f"Initializing Text2SQLHandler with model_dir={self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        texts = []

        for row in data:
            body = row.get("body") or row.get("data")

            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")

            if isinstance(body, str):
                body = json.loads(body)

            text = body.get("query") or body.get("input_text")
            if text is None:
                raise ValueError("Request body must contain 'query' or 'input_text'")

            texts.append(text)

        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        return encodings

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )
        return outputs

    def postprocess(self, inference_output):
        decoded = self.tokenizer.batch_decode(
            inference_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [{"sql": text} for text in decoded]
"""

def write_handler_file(output_dir: str):
    """
    Writes handler.py into the given directory so it can be packaged with the model.
    """
    handler_path = os.path.join(output_dir, "handler.py")
    with open(handler_path, "w", encoding="utf-8") as f:
        f.write(HANDLER_CODE)
    print(f"✅ handler.py written to: {handler_path}")

def download_from_gcs_if_needed(path):
    """
    Download file or folder from GCS if path starts with gs://
    Returns local path
    """
    if not path.startswith("gs://"):
        return path

    print(f"Downloading from GCS: {path}")
    path_parts = path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    object_path = path_parts[1] if len(path_parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Handle single file (CSV, JSON, TXT)
    if object_path.endswith((".csv", ".json", ".txt")):
        local_path = f"/tmp/{os.path.basename(object_path)}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob = bucket.blob(object_path)
        blob.download_to_filename(local_path)
        print(f"✅ Downloaded file to {local_path}")
        return local_path

    # Otherwise treat as folder
    local_dir = f"/tmp/{os.path.basename(object_path.rstrip('/'))}"
    os.makedirs(local_dir, exist_ok=True)
    blobs = bucket.list_blobs(prefix=object_path)
    for blob in blobs:
        if not blob.name.endswith("/"):
            relative_path = blob.name[len(object_path):].lstrip("/")
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
    print(f"✅ Downloaded folder to {local_dir}")
    return local_dir

def upload_to_gcs(local_path, gcs_path):
    """Uploads a local directory to GCS."""
    print(f"Uploading {local_path} to {gcs_path}...")
    client = storage.Client()
    
    # Remove gs:// prefix to get bucket and path
    path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = path_parts[0]
    gcs_prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    bucket = client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Calculate relative path to maintain structure
            relative_path = os.path.relpath(local_file_path, local_path)
            blob_path = os.path.join(gcs_prefix, relative_path)
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded: {blob_path}")

def install_torch_model_archiver():
    try:
        subprocess.check_call(["torch-model-archiver", "--version"])
        print("✅ torch-model-archiver already installed")
    except Exception:
        print("📦 Installing torch-model-archiver...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch-model-archiver", "torchserve"
        ])
        print("✅ torch-model-archiver installed successfully")

def main():
    """
    Main function to fine-tune a Hugging Face model with LoRA and upload to GCS.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model with LoRA")
    
    # Paths
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Tokenizer / model settings
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    
    # Training hyperparameters
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=25e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", nargs='+', default=["q", "v"])
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Training Configuration:")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Train data: {args.train_data}")
    print(f"  Val data: {args.val_data}")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download model from GCS if needed
    local_model_dir = download_from_gcs_if_needed(args.model_dir)
    
    # Verify model files exist
    print(f"Checking model directory: {local_model_dir}")
    if not os.path.exists(local_model_dir):
        raise ValueError(f"Model directory does not exist: {local_model_dir}")
    
    model_files = os.listdir(local_model_dir)
    print(f"Files in model directory: {model_files}")
    
    required_files = ['config.json']
    for req_file in required_files:
        if req_file not in model_files:
            raise ValueError(f"Missing required file: {req_file}")
    
    # Load datasets
    local_train_csv = download_from_gcs_if_needed(args.train_data)
    local_val_csv = download_from_gcs_if_needed(args.val_data)

    # Load datasets into pandas DataFrames
    df_train = pd.read_csv(local_train_csv)
    df_val = pd.read_csv(local_val_csv)
    
    print(f"Train dataset size: {len(df_train)}")
    print(f"Val dataset size: {len(df_val)}")
    
    # Optional: sampling for memory limits
    train_samples = args.train_samples
    val_samples = args.val_samples
    df_train = df_train.sample(n=min(len(df_train), train_samples), random_state=42)
    df_val = df_val.sample(n=min(len(df_val), val_samples), random_state=42)
    
    gc.collect()
    
    dataset_train = Dataset.from_pandas(df_train[["input_text", "sql"]])
    dataset_val = Dataset.from_pandas(df_val[["input_text", "sql"]])
    
    # Load tokenizer and model
    print(f"Loading tokenizer from: {local_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded successfully")

    print(f"Loading model from: {local_model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_model_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True
    )
    model.to(device)
    print("✅ Model loaded successfully")
    
    model.gradient_checkpointing_enable()
    tokenizer.truncation_side = "right"
    model.config.use_cache = False
    
    # Preprocessing function
    def preprocess_batch(batch):
        inputs = batch["input_text"]
        targets = batch["sql"]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, truncation=True, padding="max_length")
        labels_ids = labels["input_ids"]
        labels_ids = [[(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in seq] for seq in labels_ids]
        model_inputs["labels"] = labels_ids
        return model_inputs
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = dataset_train.map(preprocess_batch, batched=True, remove_columns=dataset_train.column_names)
    tokenized_val = dataset_val.map(preprocess_batch, batched=True, remove_columns=dataset_val.column_names)
    
    # LoRA config
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=args.target_modules
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    local_output_dir = "/tmp/lora_trained"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{local_output_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.model_accepts_loss_kwargs = False
    
    print("Starting training...")
    trainer.train()
    
    # Merge LoRA adapters into base model
    print("Merging LoRA adapter into base model...")
    merged_model_dir = "/tmp/merged_model"
    os.makedirs(merged_model_dir, exist_ok=True)
    
    # Calling .merge_and_unload() to get the merged base model.
    merged_model = model.merge_and_unload()
    print("Merge complete.")

    # Save merged model locally
    print(f"Saving merged model to: {merged_model_dir}")
    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    model_pt_path = os.path.join(merged_model_dir, "model.pt")
    torch.save(merged_model.state_dict(), model_pt_path)
    print("model.pt saved successfully")
    print(f"Merged model saved locally at: {merged_model_dir}")
    write_handler_file(merged_model_dir)

    print("\n" + "="*70)
    print("📋 DEBUGGING: Checking merged_model_dir contents")
    print("="*70)
    print(f"Directory: {merged_model_dir}")
    print(f"Exists: {os.path.exists(merged_model_dir)}")
    print(f"Is directory: {os.path.isdir(merged_model_dir)}")

    if os.path.exists(merged_model_dir):
        print("\n📁 All files in merged_model_dir:")
        all_files = sorted(os.listdir(merged_model_dir))
        for item in all_files:
            item_path = os.path.join(merged_model_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  ✓ {item:<45} {size:>15,} bytes")
            elif os.path.isdir(item_path):
                print(f"  📁 {item}/")
        
        print(f"\nTotal files: {len([f for f in all_files if os.path.isfile(os.path.join(merged_model_dir, f))])}")
        print(f"Total directories: {len([f for f in all_files if os.path.isdir(os.path.join(merged_model_dir, f))])}")
        
        # Check specific required files
        print("\n🔍 Checking critical files:")
        critical_files = [
            "model.pt",
            "handler.py", 
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json"
        ]
        
        for filename in critical_files:
            filepath = os.path.join(merged_model_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✅ {filename:<45} {size:>15,} bytes")
            else:
                print(f"  ❌ {filename:<45} NOT FOUND")
        
        # Check if model.pt is valid
        print("\n🔍 Validating model.pt:")
        model_pt_path = os.path.join(merged_model_dir, "model.pt")
        if os.path.exists(model_pt_path):
            try:
                size = os.path.getsize(model_pt_path)
                print(f"  - Size: {size:,} bytes ({size / (1024**2):.2f} MB)")
                
                # Try to load it
                test_load = torch.load(model_pt_path, map_location='cpu')
                print(f"  - Loadable: ✅ YES")
                print(f"  - Type: {type(test_load)}")
                if isinstance(test_load, dict):
                    print(f"  - Keys count: {len(test_load)}")
                del test_load  # Free memory
            except Exception as e:
                print(f"  - Loadable: ❌ NO - {e}")
        else:
            print(f"  ❌ model.pt does not exist!")
        
        # Check handler.py
        print("\n🔍 Validating handler.py:")
        handler_path = os.path.join(merged_model_dir, "handler.py")
        if os.path.exists(handler_path):
            with open(handler_path, 'r') as f:
                handler_content = f.read()
                lines = len(handler_content.split('\n'))
                print(f"  ✅ handler.py exists ({lines} lines, {len(handler_content)} chars)")
        else:
            print(f"  ❌ handler.py does not exist!")

    print("="*70 + "\n")

    if args.output_dir.startswith("gs://"):
        print(f"Merged model uploaded to GCS: {args.output_dir}")
    else:
        print(f"Warning: Output dir {args.output_dir} is not a GCS path. Data may be lost.")

    install_torch_model_archiver()

    # Build TorchServe .mar file using torch-model-archiver
    export_path = merged_model_dir

    possible_extra_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",]

    existing_extra_files = [
        f for f in possible_extra_files
        if os.path.exists(os.path.join(merged_model_dir, f))
    ]

    print(f"Extra files to include: {existing_extra_files}")

    if not existing_extra_files:
        raise FileNotFoundError("No extra files found!")

    extra_files_str = ",".join(existing_extra_files)

    cmd = [
    "torch-model-archiver",
    "--model-name", "model",
    "--version", "1.0",
    "--serialized-file", "model.pt",
    "--handler", "handler.py",
    "--extra-files", extra_files_str,
    "--export-path", ".",
    "--force",
    ]

    print(f"\n{'='*70}")
    print("🚀 Running torch-model-archiver")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {merged_model_dir}")
    print(f"Export path: {export_path}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=merged_model_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        print("✅ torch-model-archiver completed successfully!")
        
        if result.stdout:
            print(f"\n📄 STDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\n📄 STDERR:\n{result.stderr}")
        
        # Verify .mar file
        mar_file = os.path.join(merged_model_dir, "model.mar")
        if os.path.exists(mar_file):
            mar_size = os.path.getsize(mar_file)
            print(f"\n✅ .mar file created: {mar_file}")
            print(f"   Size: {mar_size:,} bytes ({mar_size / (1024**2):.2f} MB)")
        else:
            print(f"\n⚠️ Warning: .mar file not found at {mar_file}")
            print("\n📁 Files in merged_model_dir after archiver:")
            for f in os.listdir(merged_model_dir):
                print(f"  - {f}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*70}")
        print("❌ TORCH-MODEL-ARCHIVER FAILED")
        print(f"{'='*70}")
        print(f"Exit code: {e.returncode}")
        print(f"\n📄 STDOUT:\n{e.stdout if e.stdout else '(empty)'}")
        print(f"\n📄 STDERR:\n{e.stderr if e.stderr else '(empty)'}")
        print(f"{'='*70}\n")
        raise

    except subprocess.TimeoutExpired:
        print("❌ torch-model-archiver timed out after 10 minutes")
        raise


    # Upload merged model to GCS
    if args.output_dir.startswith("gs://"):
        upload_to_gcs(merged_model_dir, args.output_dir)
        print(f"Merged model uploaded to GCS: {args.output_dir}")
    else:
        print(f"Warning: Output dir {args.output_dir} is not a GCS path. Data may be lost.")

    print("Training and upload completed.")

if __name__ == "__main__":
    main()