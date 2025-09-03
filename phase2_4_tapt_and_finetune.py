# phase2_4_tapt_and_finetune.py
# Phase 2.4 — TAPT (MLM) + supervised fine-tuning for sentiment classification.
# Writes ALL artifacts under results/phase2/phase2_4/.
# M1 (8GB) friendly: small batch + grad accumulation + gradient checkpointing + memory cleanup + PARTIAL FINE-TUNING.

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import gc, torch

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from datasets import load_dataset, DatasetDict

from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# ---------------------------
# Utilities
# ---------------------------

def get_device_str():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def detect_text_column(columns: List[str]) -> str:
    preferred = ["text_bert", "text", "review_text", "review", "content", "body"]
    for c in preferred:
        if c in columns:
            return c
    # Fallback: first non-label-like column
    for c in columns:
        if c.lower() not in {"label", "labels", "sentiment", "target", "y"}:
            return c
    raise ValueError("No suitable text column found. Set --text_column.")

def build_label_mapping(series: pd.Series) -> Dict[str, int]:
    # If already numeric 0..K-1, keep as-is
    if pd.api.types.is_integer_dtype(series) and series.min() >= 0 and series.nunique() <= 20:
        classes = sorted(series.unique().tolist())
        return {str(c): int(c) for c in classes}
    # Otherwise, treat as string labels
    classes = sorted(series.astype(str).unique().tolist())
    return {cls: i for i, cls in enumerate(classes)}

def apply_label_mapping(series: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    return series.astype(str).map(mapping).values

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_png: str):
    fig = plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---------------------------
# Tokenize helpers
# ---------------------------

def tokenize_for_mlm(tokenizer, text_col, max_len):
    def fn(batch: Dict[str, Any]):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
            padding=False,                  # dynamic padding via data collator
            return_token_type_ids=False,    # memory-friendly
        )
    return fn

def tokenize_for_cls(tokenizer, text_col, max_len):
    def fn(batch: Dict[str, Any]):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
            padding=False,
            return_token_type_ids=False,    # memory-friendly
        )
    return fn

# ---------------------------
# Partial FT helper
# ---------------------------

def apply_partial_finetune(model: BertForSequenceClassification, unfreeze_last_n: int):
    """
    Freeze all BERT layers, then unfreeze last N encoder blocks + always keep classifier trainable.
    """
    # freeze everything
    for p in model.bert.parameters():
        p.requires_grad = False

    # unfreeze last N encoder layers
    if unfreeze_last_n > 0:
        enc = model.bert.encoder
        for layer in enc.layer[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # keep pooler trainable if present
    if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
        for p in model.bert.pooler.parameters():
            p.requires_grad = True

    # classifier is always trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    print(f"[FT] Partial fine-tuning applied: last {unfreeze_last_n} encoder block(s) + classifier are trainable.")

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2.4 — TAPT (MLM) + fine-tuning")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/val.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="sentiment")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")

    # TAPT (MLM) params
    parser.add_argument("--tapt_epochs", type=int, default=1)
    parser.add_argument("--tapt_max_len", type=int, default=64)     # conservative default
    parser.add_argument("--tapt_batch", type=int, default=2)
    parser.add_argument("--tapt_grad_accum", type=int, default=16)
    parser.add_argument("--tapt_lr", type=float, default=1e-4)
    parser.add_argument("--tapt_warmup", type=float, default=0.10)
    parser.add_argument("--tapt_mlm_prob", type=float, default=0.15)

    # Fine-tune params (UPDATED defaults per thesis plan)
    parser.add_argument("--ft_epochs", type=int, default=4)          # was 3
    parser.add_argument("--ft_max_len", type=int, default=96)        # was 128
    parser.add_argument("--ft_batch", type=int, default=4)           # was 8
    parser.add_argument("--ft_grad_accum", type=int, default=8)      # was 4
    parser.add_argument("--ft_lr", type=float, default=3e-5)         # was 2e-5
    parser.add_argument("--ft_warmup", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)

    # Partial fine-tuning flag (UPDATED default)
    parser.add_argument("--unfreeze_last_n", type=int, default=1,
                        help="How many last transformer layers to unfreeze (0 = freeze all except classifier)")

    args = parser.parse_args()

    # Paths
    ROOT_OUT = "results/phase2/phase2_4"
    TAPT_OUT = os.path.join(ROOT_OUT, "tapt_mlm")
    FT_OUT   = os.path.join(ROOT_OUT, "finetune_cls")
    ensure_dir(ROOT_OUT); ensure_dir(TAPT_OUT); ensure_dir(FT_OUT)

    set_seed(args.seed)
    device = get_device_str()
    print(f"[INFO] Device: {device}")

    # ---------------------------
    # Load CSVs
    # ---------------------------
    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)
    df_test  = pd.read_csv(args.test_csv)

    text_col = args.text_column or detect_text_column(df_train.columns.tolist())
    if text_col not in df_train.columns:
        raise ValueError(f"Text column '{text_col}' not in train CSV.")
    if args.label_column not in df_train.columns:
        raise ValueError(f"Label column '{args.label_column}' not in train CSV.")

    # Label mapping (fit on train only)
    label_map = build_label_mapping(df_train[args.label_column])
    with open(os.path.join(ROOT_OUT, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # ---------------------------
    # TAPT (MLM) on TRAIN TEXT ONLY (no labels)
    # ---------------------------
    print("[TAPT] Starting task-adaptive pretraining (MLM) on train text only...")

    ds_train_only = load_dataset("csv", data_files={"train": args.train_csv})
    cols = ds_train_only["train"].column_names
    to_remove = [c for c in cols if c != text_col]
    ds_train_only = ds_train_only.remove_columns(to_remove)

    tokenizer_mlm = BertTokenizerFast.from_pretrained(args.base_model)
    model_mlm = BertForMaskedLM.from_pretrained(args.base_model)
    model_mlm.gradient_checkpointing_enable()

    ds_tok_mlm = ds_train_only.map(
        tokenize_for_mlm(tokenizer_mlm, text_col, args.tapt_max_len),
        batched=True
    )

    collator_mlm = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_mlm, mlm=True, mlm_probability=args.tapt_mlm_prob
    )

    tapt_args = TrainingArguments(
        output_dir=TAPT_OUT,
        overwrite_output_dir=True,
        num_train_epochs=args.tapt_epochs,
        per_device_train_batch_size=args.tapt_batch,
        gradient_accumulation_steps=args.tapt_grad_accum,
        learning_rate=args.tapt_lr,
        weight_decay=0.01,
        warmup_ratio=args.tapt_warmup,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        group_by_length=True,
        seed=args.seed,
    )

    trainer_mlm = Trainer(
        model=model_mlm,
        args=tapt_args,
        train_dataset=ds_tok_mlm["train"],
        tokenizer=tokenizer_mlm,
        data_collator=collator_mlm,
    )

    trainer_mlm.train()
    trainer_mlm.save_model(TAPT_OUT)
    tokenizer_mlm.save_pretrained(TAPT_OUT)
    print(f"[TAPT] Saved TAPT checkpoint to {TAPT_OUT}")

    # Free memory BEFORE fine-tuning
    del trainer_mlm, model_mlm, tokenizer_mlm, ds_tok_mlm, ds_train_only
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # ---------------------------
    # Prepare supervised datasets (train/val/test)
    # ---------------------------
    def map_labels(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label"] = apply_label_mapping(out[args.label_column], label_map)
        return out[[text_col, "label"]]

    sup_train = map_labels(df_train)
    sup_val   = map_labels(df_val)
    sup_test  = map_labels(df_test)

    dataset = DatasetDict({
        "train": load_dataset("csv", data_files={"train": args.train_csv})["train"],
        "val":   load_dataset("csv", data_files={"val": args.val_csv})["val"],
        "test":  load_dataset("csv", data_files={"test": args.test_csv})["test"],
    })

    # keep only text + add mapped labels
    dataset["train"] = dataset["train"].remove_columns([c for c in dataset["train"].column_names if c not in [text_col]])
    dataset["val"]   = dataset["val"].remove_columns([c for c in dataset["val"].column_names   if c not in [text_col]])
    dataset["test"]  = dataset["test"].remove_columns([c for c in dataset["test"].column_names  if c not in [text_col]])

    dataset["train"] = dataset["train"].add_column("label", sup_train["label"].tolist())
    dataset["val"]   = dataset["val"].add_column("label", sup_val["label"].tolist())
    dataset["test"]  = dataset["test"].add_column("label", sup_test["label"].tolist())

    # Tokenizer / Model for classification (from TAPT ckpt)
    tokenizer_cls = BertTokenizerFast.from_pretrained(TAPT_OUT)
    model_cls = BertForSequenceClassification.from_pretrained(TAPT_OUT, num_labels=len(label_map))
    model_cls.gradient_checkpointing_enable()

    # Partial FT (unfreeze last N)
    apply_partial_finetune(model_cls, unfreeze_last_n=args.unfreeze_last_n)

    ds_tok_train = dataset["train"].map(tokenize_for_cls(tokenizer_cls, text_col, args.ft_max_len), batched=True)
    ds_tok_val   = dataset["val"].map(tokenize_for_cls(tokenizer_cls, text_col, args.ft_max_len), batched=True)
    ds_tok_test  = dataset["test"].map(tokenize_for_cls(tokenizer_cls, text_col, args.ft_max_len), batched=True)

    collator_cls = DataCollatorWithPadding(tokenizer=tokenizer_cls)

    ft_args = TrainingArguments(
        output_dir=FT_OUT,
        overwrite_output_dir=True,
        num_train_epochs=args.ft_epochs,
        per_device_train_batch_size=args.ft_batch,
        gradient_accumulation_steps=args.ft_grad_accum,
        learning_rate=args.ft_lr,
        weight_decay=0.01,
        warmup_ratio=args.ft_warmup,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",  # CHANGED
        greater_is_better=True,                 # CHANGED
        logging_steps=100,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        group_by_length=True,
        eval_accumulation_steps=32,  # reduce eval memory
        max_grad_norm=1.0,           # stabilize
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    trainer_cls = Trainer(
        model=model_cls,
        args=ft_args,
        train_dataset=ds_tok_train,
        eval_dataset=ds_tok_val,
        tokenizer=tokenizer_cls,
        data_collator=collator_cls,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[FT] Supervised fine-tuning starting...")
    trainer_cls.train()
    trainer_cls.save_model(FT_OUT)
    tokenizer_cls.save_pretrained(FT_OUT)
    print(f"[FT] Saved fine-tuned classifier to {FT_OUT}")

    # ---------------------------
    # Evaluation on TEST
    # ---------------------------
    print("[EVAL] Running test evaluation...")
    preds_raw = trainer_cls.predict(ds_tok_test)
    y_true = np.array(ds_tok_test["label"])
    y_pred = np.argmax(preds_raw.predictions, axis=1)

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in range(len(inv_map))]

    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(ROOT_OUT, "classification_report.csv"))

    acc = (y_true == y_pred).mean()
    macro_f1 = report_df.loc["macro avg", "f1-score"]
    macro_prec = report_df.loc["macro avg", "precision"]
    macro_rec = report_df.loc["macro avg", "recall"]

    metrics = pd.DataFrame([{
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
    }])
    metrics.to_csv(os.path.join(ROOT_OUT, "test_metrics.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    plot_confusion_matrix(cm, target_names, os.path.join(ROOT_OUT, "confusion_matrix.png"))

    preds_df = pd.DataFrame({
        "id": np.arange(len(y_true)),
        "true_label": [inv_map[int(i)] for i in y_true],
        "pred_label": [inv_map[int(i)] for i in y_pred],
    })
    preds_df.to_csv(os.path.join(ROOT_OUT, "predictions.csv"), index=False)

    print(f"[DONE] Test metrics saved under: {ROOT_OUT}")
    print(f"[DONE] Files: test_metrics.csv, classification_report.csv, confusion_matrix.png, predictions.csv, label_mapping.json")

if __name__ == "__main__":
    main()
