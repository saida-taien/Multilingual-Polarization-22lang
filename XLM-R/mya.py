# ============================================================
# XLM-RoBERTa-base | SemEval 2026 Task 9 | TEST PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import torch
from google.colab import files
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ---------------- CONFIG ----------------
LANG = "mya"   # change language
MODEL_NAME = "xlm-roberta-base"

TRAIN_PATH = f"/content/drive/MyDrive/Colab Notebooks/SemEval2026Task-9/Subtask-01/Datasets/TrainSet/{LANG}.csv"
TEST_PATH  = f"/content/drive/MyDrive/Colab Notebooks/SemEval2026Task-9/Subtask-01/Datasets/TestSet/{LANG}.csv"

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

def clean_text(x):
    return str(x).replace("\n", " ").strip()

train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"]  = test_df["text"].apply(clean_text)

# ---------------- TRAIN / VAL SPLIT ----------------
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["polarization"],
    random_state=42
)

train_ds = Dataset.from_pandas(train_df[["text","polarization"]])
val_ds   = Dataset.from_pandas(val_df[["text","polarization"]])
test_ds  = Dataset.from_pandas(test_df[["id","text"]])

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("polarization","labels")
val_ds   = val_ds.rename_column("polarization","labels")

train_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])
val_ds.set_format("torch", columns=["input_ids","attention_mask","labels"])
test_ds.set_format("torch", columns=["input_ids","attention_mask"])

# ---------------- MODEL ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

args = TrainingArguments(
    output_dir="./xlm_results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()

# ---------------- TEST PREDICTION ----------------
preds = trainer.predict(test_ds)
probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()

prob_df = pd.DataFrame({
    "id": test_df["id"],
    "prob_0": probs[:,0],
    "prob_1": probs[:,1]
})

file_name = f"pred_{LANG}_xlm_probs.csv"
prob_df.to_csv(file_name, index=False)
files.download(file_name)
print("DONE")
