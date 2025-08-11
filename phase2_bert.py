# phase2_bert.py
# Phase 2.0 — Baseline BERT (Frozen encoder, feature-based)
# - Extract [CLS] embeddings with bert-base-uncased (no domain fine-tuning)
# - Train Logistic Regression on top of frozen embeddings
# - Save reports/CSVs/graphs mirroring Phase 1 (VADER)

import os
import re
import json
import time
import math
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_WORKERS = 2

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL2ID = {l:i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamped_outdir(phase_stub: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", "phase2", phase_stub, ts)
    ensure_dir(outdir)
    return outdir, ts

def load_splits():
    train_df = pd.read_csv("data/train.csv")
    val_df   = pd.read_csv("data/val.csv")
    test_df  = pd.read_csv("data/test.csv")
    return train_df, val_df, test_df

def prepare_text_and_labels(df: pd.DataFrame, text_col="text_bert", label_col="sentiment"):
    if text_col not in df.columns:
        # fallback
        text_col = "text_raw" if "text_raw" in df.columns else df.columns[0]
    texts = df[text_col].astype(str).fillna("").tolist()
    labels = df[label_col].astype(str).tolist()
    y = np.array([LABEL2ID[l] for l in labels])
    return texts, y

# -----------------------------
# Dataset for embedding extraction
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze batch dim
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item

# -----------------------------
# Embedding extraction ([CLS])
# -----------------------------
@torch.no_grad()
def extract_cls_embeddings(texts, tokenizer, model, device):
    ds = TextDataset(texts, tokenizer, MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    reps = []
    model.eval()
    for batch in tqdm(dl, desc="Embedding"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # last_hidden_state: (bs, seq_len, hidden)
        # [CLS] token is position 0
        cls_vec = outputs.last_hidden_state[:, 0, :]  # (bs, hidden)
        reps.append(cls_vec.cpu().numpy())
    reps = np.vstack(reps)
    return reps  # shape: (N, hidden_size)

# -----------------------------
# Metrics & reporting
# -----------------------------
def compute_overview_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted")
    }

def class_summary(y_true, y_pred):
    labels = list(range(len(LABEL_ORDER)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, lab in enumerate(LABEL_ORDER):
        total = cm[i].sum()
        correct = cm[i, i]
        acc_cls = correct / total if total > 0 else 0.0
        rows.append([lab, int(total), int(correct), acc_cls])
    rows.append([
        "TOTAL", int(cm.sum()), int((y_true == y_pred).sum()),
        accuracy_score(y_true, y_pred)
    ])
    return cm, pd.DataFrame(rows, columns=["Class", "Total", "Correct", "Accuracy per Class"])

def save_reports(outdir, split_name, y_true, y_pred, proba=None):
    # Overview metrics
    ov = compute_overview_metrics(y_true, y_pred)
    ov_df = pd.DataFrame([[split_name, *ov.values()]],
                         columns=["Dataset", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"])
    ov_path = os.path.join(outdir, f"bert_overview_metrics_{split_name}.csv")
    ov_df.to_csv(ov_path, index=False)

    # Classification report
    cr = classification_report(y_true, y_pred, target_names=LABEL_ORDER, output_dict=True, digits=4)
    cr_df = pd.DataFrame(cr).transpose()
    cr_path = os.path.join(outdir, f"bert_{split_name}_report.csv")
    cr_df.to_csv(cr_path)

    # Class summary + confusion matrix
    cm, cs_df = class_summary(y_true, y_pred)
    cs_path = os.path.join(outdir, f"bert_class_summary_{split_name}.csv")
    cs_df.to_csv(cs_path, index=False)

    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)
    cm_path = os.path.join(outdir, f"bert_confusion_matrix_{split_name}.csv")
    cm_df.to_csv(cm_path)

    # Predictions CSV (optionally with probabilities)
    pred_labels = [ID2LABEL[i] for i in y_pred]
    pred_df = pd.DataFrame({"true": [ID2LABEL[i] for i in y_true], "pred": pred_labels})
    if proba is not None:
        for i, lab in enumerate(LABEL_ORDER):
            pred_df[f"proba_{lab}"] = proba[:, i]
    pred_path = os.path.join(outdir, f"bert_predictions_{split_name}.csv")
    pred_df.to_csv(pred_path, index=False)

    # Graphs
    # 1) Confusion matrix heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
    plt.title(f"Confusion Matrix – BERT ({split_name})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bert_confusion_matrix_{split_name}.png"), dpi=300)
    plt.close()

    # 2) Correct vs Incorrect per class
    correct_counts = np.diag(cm)
    incorrect_counts = cm.sum(axis=1) - correct_counts
    x = np.arange(len(LABEL_ORDER))
    plt.figure(figsize=(8,5))
    plt.bar(x - 0.2, correct_counts, width=0.4, label="Correct")
    plt.bar(x + 0.2, incorrect_counts, width=0.4, label="Incorrect")
    plt.xticks(x, LABEL_ORDER)
    plt.title(f"Correct vs Incorrect per Class – BERT ({split_name})")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bert_correct_incorrect_{split_name}.png"), dpi=300)
    plt.close()

    return ov_df, cr_df, cs_df, cm_df

# -----------------------------
# Phase 2.0 – Baseline (Frozen Encoder)
# -----------------------------
def run_phase_20_baseline(device="cuda" if torch.cuda.is_available() else "cpu"):
    outdir, ts = timestamped_outdir("phase2_0_baseline")
    print(f"[Phase 2.0] Output -> {outdir}")

    # Load data
    train_df, val_df, test_df = load_splits()
    X_train, y_train = prepare_text_and_labels(train_df, text_col="text_bert")
    X_val,   y_val   = prepare_text_and_labels(val_df,   text_col="text_bert")
    X_test,  y_test  = prepare_text_and_labels(test_df,  text_col="text_bert")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # Extract [CLS] embeddings (frozen encoder)
    print("Extracting embeddings (train)...")
    train_reps = extract_cls_embeddings(X_train, tokenizer, model, device)
    print("Extracting embeddings (val)...")
    val_reps   = extract_cls_embeddings(X_val, tokenizer, model, device)
    print("Extracting embeddings (test)...")
    test_reps  = extract_cls_embeddings(X_test, tokenizer, model, device)

    # Train linear classifier
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    clf.fit(train_reps, y_train)

    # Evaluate
    y_train_pred = clf.predict(train_reps)
    y_val_pred   = clf.predict(val_reps)
    y_test_pred  = clf.predict(test_reps)

    # Predict probabilities (for CSV)
    train_proba = clf.predict_proba(train_reps)
    val_proba   = clf.predict_proba(val_reps)
    test_proba  = clf.predict_proba(test_reps)

    # Save reports
    train_ov, train_cr, train_cs, train_cm = save_reports(outdir, "train", y_train, y_train_pred, train_proba)
    val_ov,   val_cr,   val_cs,   val_cm   = save_reports(outdir, "val",   y_val,   y_val_pred,   val_proba)
    test_ov,  test_cr,  test_cs,  test_cm  = save_reports(outdir, "test",  y_test,  y_test_pred,  test_proba)

    # Phase summary (append)
    phase_name = "Phase_2_0_Baseline_FrozenBERT"
    phase_summary_file = "results/phase_summary.csv"
    phase_overview_df = pd.DataFrame([
        [phase_name, "Train", *train_ov.iloc[0,1:].tolist()],
        [phase_name, "Val",   *val_ov.iloc[0,1:].tolist()],
        [phase_name, "Test",  *test_ov.iloc[0,1:].tolist()],
    ], columns=["Phase", "Dataset", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"])

    if os.path.exists(phase_summary_file):
        old_df = pd.read_csv(phase_summary_file)
        combined_df = pd.concat([old_df, phase_overview_df], ignore_index=True)
        combined_df.to_csv(phase_summary_file, index=False)
    else:
        phase_overview_df.to_csv(phase_summary_file, index=False)

    print(f"[Phase 2.0] Done. Outputs saved under: {outdir}")

# -----------------------------
# Stubs for next phases (to be implemented)
# -----------------------------
def run_phase_21_finetune():
    """
    Phase 2.1 — Domain-Adaptive Fine-Tuning
    - To be implemented: full fine-tuning with HuggingFace Trainer on train set, eval on val/test
    - Save metrics/CSVs/plots mirroring Phase 1 & Phase 2.0
    """
    raise NotImplementedError("Phase 2.1 not implemented yet.")

def run_phase_22_kfold():
    """
    Phase 2.2 — K-Fold Cross Validation (e.g., 5-fold)
    - To be implemented: repeat fine-tuning across folds, average metrics
    """
    raise NotImplementedError("Phase 2.2 not implemented yet.")

def run_phase_23_hparam():
    """
    Phase 2.3 — Hyperparameter Optimisation
    - To be implemented: grid/Bayesian search over LR, batch size, max_len, epochs
    """
    raise NotImplementedError("Phase 2.3 not implemented yet.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="2.0", help="Which Phase 2 step to run: 2.0 | 2.1 | 2.2 | 2.3")
    args = parser.parse_args()

    if args.phase == "2.0":
        run_phase_20_baseline()
    elif args.phase == "2.1":
        run_phase_21_finetune()
    elif args.phase == "2.2":
        run_phase_22_kfold()
    elif args.phase == "2.3":
        run_phase_23_hparam()
    else:
        raise ValueError("Unknown phase argument. Use one of: 2.0 | 2.1 | 2.2 | 2.3")