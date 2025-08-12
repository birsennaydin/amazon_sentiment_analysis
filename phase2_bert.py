import sys
print(sys.modules.get("transformers"))
import transformers
print("Transformers version (runtime):", transformers.__version__)
from transformers import TrainingArguments
print("TrainingArguments source:", TrainingArguments.__module__)
import inspect
print("TrainingArguments file:", inspect.getfile(TrainingArguments))

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, set_seed
)

# -----------------------------
# 0) Çıkış klasörü ve tohum
# -----------------------------
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"results/phase2/phase2_1_domain_finetune/{STAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(42)
print(f"[Phase 2.1] Output -> {OUT_DIR}")

# -----------------------------
# 1) Veriyi yükle
# -----------------------------
CSV_PATH = "data/amazon_electronics_prepared.csv"
df = pd.read_csv(CSV_PATH)

assert "text_bert" in df.columns and "sentiment" in df.columns, \
    "CSV içinde 'text_bert' ve 'sentiment' kolonları olmalı."

label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}
df = df[df["sentiment"].isin(label2id.keys())].copy()
df["label"] = df["sentiment"].map(label2id)

# -----------------------------
# 2) Stratified bölme
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    df["text_bert"], df["label"],
    test_size=0.30, stratify=df["label"], random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# -----------------------------
# 3) Tokenizer ve encode
# -----------------------------
MAX_LEN = 128
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def encode_texts(texts: List[str]):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

enc_train = encode_texts(X_train)
enc_val   = encode_texts(X_val)
enc_test  = encode_texts(X_test)

@dataclass
class HFDataset(torch.utils.data.Dataset):
    encodings: Dict[str, List[int]]
    labels: List[int]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = HFDataset(enc_train, list(y_train))
val_ds   = HFDataset(enc_val,   list(y_val))
test_ds  = HFDataset(enc_test,  list(y_test))

# -----------------------------
# 4) Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# -----------------------------
# 5) Metrikler
# -----------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# -----------------------------
# 6) TrainingArguments & Trainer
# -----------------------------
use_fp16 = torch.cuda.is_available()

args = TrainingArguments(
    output_dir=str(OUT_DIR / "checkpoints"),
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_dir=str(OUT_DIR / "logs"),
    logging_steps=50,
    eval_strategy="epoch",   # DÜZELTİLDİ — eski 'evaluation_strategy' yerine
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=use_fp16,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# 7) Fine-tuning
# -----------------------------
trainer.train()

# -----------------------------
# 8) Test değerlendirme
# -----------------------------
test_metrics = trainer.evaluate(test_ds)
print("[Phase 2.1] Test metrics:", test_metrics)
pd.DataFrame([test_metrics]).to_csv(OUT_DIR / "metrics_test_overall.csv", index=False)

# Sınıf bazlı rapor & karışıklık matrisi
pred_outputs = trainer.predict(test_ds)
test_preds = np.argmax(pred_outputs.predictions, axis=1)
test_probs = torch.softmax(torch.tensor(pred_outputs.predictions), dim=1).numpy()

cm = confusion_matrix(y_test, test_preds, labels=[0,1,2])
pd.DataFrame(cm, index=[id2label[i] for i in [0,1,2]],
             columns=[id2label[i] for i in [0,1,2]]).to_csv(OUT_DIR / "confusion_matrix.csv")

rep = classification_report(y_test, test_preds, target_names=[id2label[i] for i in [0,1,2]], output_dict=True, zero_division=0)
pd.DataFrame(rep).transpose().to_csv(OUT_DIR / "classification_report.csv")

pred_df = pd.DataFrame({
    "text": list(X_test),
    "gold": [id2label[i] for i in y_test],
    "pred": [id2label[i] for i in test_preds],
    "prob_positive": test_probs[:,0],
    "prob_neutral":  test_probs[:,1],
    "prob_negative": test_probs[:,2],
})
pred_df.to_csv(OUT_DIR / "predictions_test.csv", index=False)

# Görsel
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix – Phase 2.1")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks([0,1,2], [id2label[i] for i in [0,1,2]])
plt.yticks([0,1,2], [id2label[i] for i in [0,1,2]])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

print(f"[Phase 2.1] Done. Outputs saved under: {OUT_DIR}")
