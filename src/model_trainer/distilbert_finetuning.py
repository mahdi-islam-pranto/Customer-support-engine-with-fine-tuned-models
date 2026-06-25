import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
import evaluate
import mlflow
import mlflow.pytorch


# ==========================================
# MLflow Setup
# ==========================================
print("📊 Setting up MLflow experiment tracking...")
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Save MLflow data locally (can use remote server)
mlflow.set_experiment("distilbert-intent-classifier")
mlflow.start_run(run_name="distilbert_training_run")

# ==========================================
# 2. LOAD AND PREPARE DATASETS
# ==========================================
print("🔄 Loading CSV datasets into Hugging Face Dataset format...")
# Load the CSV files we uploaded into Pandas DataFrames
train_df = pd.read_csv("data/intent_train.csv")
test_df = pd.read_csv("data/intent_test.csv")

# DeBERTa expects the text column to be named 'text' and label to be named 'label'
# (Our previous script already formatted them as 'text' and 'label')
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine them into a single DatasetDict for clean mapping
raw_datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})


# ==========================================
# 3. TOKENIZATION
# ==========================================
MODEL_CKPT = "distilbert-base-uncased"
print(f"📥 Loading tokenizer for {MODEL_CKPT}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)

def tokenize_function(examples):
    # Truncate queries longer than 128 tokens. Customer support inputs are usually short,
    # so keeping max_length small saves a massive amount of GPU memory and training time.
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("⚡ Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Data collator dynamically pads the batches to the maximum length in that specific batch,
# which is much faster than padding everything to a static 128 length.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ==========================================
# 4. DEFINE EVALUATION METRICS
# ==========================================
# Load accuracy and f1 metrics to monitor performance during training epochs
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert model logits (raw outputs) into class predictions (0 or 1) via argmax
    preds = np.argmax(predictions, axis=1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}


# ==========================================
# 5. INITIALIZE THE MODEL
# ==========================================
print(f"🤖 Initializing model: {MODEL_CKPT} for Binary Classification...")
# num_labels=2 sets up the sequence classification head with 2 output nodes
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=2)


# ==========================================
# 6. CONFIGURING TRAINING ARGUMENTS (STABILIZED)
# ==========================================
# Log hyperparameters to MLflow
hyperparams = {
    "model_checkpoint": MODEL_CKPT,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "eval_strategy": "epoch",
    "max_length": 128
}
mlflow.log_params(hyperparams)
print(f"✅ Logged {len(hyperparams)} hyperparameters to MLflow")

training_args = TrainingArguments(
    output_dir="./deberta-intent-router",      # Directory where checkpoints are saved
    learning_rate=2e-5,                        # Standard stable learning rate for DeBERTa
    per_device_train_batch_size=16,            # Fits comfortably inside T4 memory
    per_device_eval_batch_size=16,
    num_train_epochs=3,                        # 3 epochs is ideal
    weight_decay=0.01,                         # Regularization technique to prevent overfitting
    eval_strategy="epoch",               # Evaluate loss/metrics at the end of every epoch
    save_strategy="epoch",                     # Save a checkpoint at the end of every epoch
    load_best_model_at_end=True,               # Keep the checkpoint that performed best
    metric_for_best_model="f1",                # Optimize for the highest F1-score
    fp16=True,                                # 🔴 CHANGED TO FALSE: Solves the DeBERTa NaN overflow bug
    max_grad_norm=1.0,                         # 🟢 ADDED: Prevents gradient explosion by clipping them at 1.0
    report_to="mlflow",                        # 🟢 ADDED: Enable MLflow logging
    logging_dir="./logs",                      # Directory for tensorboard logs
    logging_steps=50                           # Log metrics every 50 steps
)

# Pass everything into the Hugging Face Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ==========================================
# 7. EXECUTE FINE-TUNING
# ==========================================
print("🚀 Starting fine-tuning loop...")
trainer.train()

print("\n🎯 Evaluating final model on test split...")
evaluation_results = trainer.evaluate()
print(f"Final Test Accuracy: {evaluation_results['eval_accuracy']:.4f}")
print(f"Final Test F1-Score: {evaluation_results['eval_f1']:.4f}")

# Log final evaluation metrics to MLflow
mlflow.log_metrics({
    "final_test_accuracy": evaluation_results['eval_accuracy'],
    "final_test_f1": evaluation_results['eval_f1'],
    "final_test_loss": evaluation_results['eval_loss']
})
print("✅ Logged final evaluation metrics to MLflow")


# ==========================================
# 8. SAVE THE FINE-TUNED WEIGHTS
# ==========================================
# This saves the model weights, configuration, and tokenizer vocabulary locally in Colab
final_model_path = "./fine_tuned_distilbert_router"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"💾 Model and tokenizer successfully saved to: {final_model_path}")

# Log model artifacts to MLflow
mlflow.pytorch.log_model(model, "distilbert_model")
mlflow.log_artifacts(final_model_path, artifact_path="model_artifacts")
print("✅ Logged model artifacts to MLflow")

# ==========================================
# 9. LOCAL PIPELINE TESTING (VERIFICATION)
# ==========================================
print("\n🔍 Verification testing on sample prompts:")
# Create an end-to-end classification pipeline using our trained model
classifier = pipeline("text-classification", model=final_model_path, tokenizer=final_model_path, device=0)

test_prompts = [
    "Where is my package?",            # Expected: LABEL_0 (Informative)
    "Cancel my package right now"      # Expected: LABEL_1 (Actionable)
]

for prompt in test_prompts:
    result = classifier(prompt)[0]
    print(f"Prompt: '{prompt}' ➡️ Prediction: {result['label']} (Confidence: {result['score']:.4f})")

# ==========================================
# 10. END MLFLOW RUN
# ==========================================
mlflow.end_run()
print("\n✅ MLflow run completed! View results with: mlflow ui")