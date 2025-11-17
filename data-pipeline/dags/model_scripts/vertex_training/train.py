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

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model with LoRA")
    
    # Paths
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Tokenizer / model settings
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    # local_model_dir = download_from_gcs_if_needed(args.model_dir)
    local_model_dir = args.model_dir
    
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
    print("Loading training data...")
    df_train = pd.read_csv(args.train_data)
    df_val = pd.read_csv(args.val_data)
    
    print(f"Train dataset size: {len(df_train)}")
    print(f"Val dataset size: {len(df_val)}")
    
    # Optional: sampling for memory limits
    df_train = df_train.sample(n=min(len(df_train), 1), random_state=42)
    df_val = df_val.sample(n=min(len(df_val), 1), random_state=42)
    
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
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
    
    # Save model
    print(f"Saving model to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Training finished. Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()