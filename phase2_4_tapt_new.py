# phase2_4_tapt_new.py
# Phase 2.4 — TAPT + Fine-tune (val selection) + Refit on Train+Val + Single-shot Test
# Saves thesis-ready tables and figures.

import os, re, json, random
from datetime import datetime
from glob import glob

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Apple MPS OOM'u azaltır

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,
    DataCollatorWithPadding, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

# -------------------- Config & paths --------------------
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--train_csv", default="data/train.csv")
ap.add_argument("--val_csv",   default="data/val.csv")
ap.add_argument("--test_csv",  default="data/test.csv")

ap.add_argument("--base_model", default="bert-base-uncased")

# TAPT (MLM) defaults — tezinizde paylaştıklarınız:
ap.add_argument("--tapt_max_len", type=int, default=96)
ap.add_argument("--tapt_lr", type=float, default=1e-4)
ap.add_argument("--tapt_epochs", type=int, default=1)
ap.add_argument("--tapt_batch", type=int, default=4)
ap.add_argument("--tapt_grad_accum", type=int, default=8)
ap.add_argument("--tapt_wd", type=float, default=0.01)
ap.add_argument("--tapt_warmup", type=float, default=0.10)
ap.add_argument("--tapt_mlm_prob", type=float, default=0.15)

# Supervised fine-tune defaults — tezdeki Phase 2.4:
ap.add_argument("--ft_max_len", type=int, default=96)
ap.add_argument("--ft_lr", type=float, default=3e-5)
ap.add_argument("--ft_epochs", type=int, default=4)
ap.add_argument("--ft_batch", type=int, default=4)
ap.add_argument("--ft_grad_accum", type=int, default=8)
ap.add_argument("--ft_wd", type=float, default=0.01)
ap.add_argument("--ft_patience", type=int, default=2)

cli = ap.parse_args()

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ROOT = "results/phase2/phase2_4_new_test"
os.makedirs(ROOT, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(ROOT, STAMP)
os.makedirs(RUN_DIR, exist_ok=True)

# -------------------- Utilities --------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "text_bert" in df.columns and "sentiment" in df.columns, "CSV must have text_bert & sentiment"
    df = df.dropna(subset=["text_bert","sentiment"]).copy()
    df["label"] = df["sentiment"].astype(str).str.lower().map(LABEL2ID)
    df = df[df["label"].isin([0,1,2])].reset_index(drop=True)
    return df

def build_cls_dataset(df: pd.DataFrame, tokenizer, max_len: int):
    def tok(batch): return tokenizer(batch["text_bert"], truncation=True, max_length=max_len)
    ds = Dataset.from_pandas(df[["text_bert","label"]].rename(columns={"label":"labels"}), preserve_index=False)
    return ds.map(tok, batched=True, remove_columns=["text_bert"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "macro_precision": p_m, "macro_recall": r_m, "macro_f1": f_m, "weighted_f1": f_w}

def latest_ckpt(path):
    xs = glob(os.path.join(path, "checkpoint-*"))
    if not xs: return None
    xs.sort(key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    return xs[-1]

def save_cm_png_csv(y_true, y_pred, out_csv, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_df = pd.DataFrame(cm, index=[ID2LABEL[i] for i in [0,1,2]],
                         columns=[ID2LABEL[i] for i in [0,1,2]])
    cm_df.to_csv(out_csv)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix — BERT (Phase 2.4)")
        plt.xticks(ticks=range(3), labels=list(cm_df.columns))
        plt.yticks(ticks=range(3), labels=list(cm_df.index))
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[INFO] CM PNG skipped: {e}")

# -------------------- Load data --------------------
train_df = load_df(cli.train_csv)
val_df   = load_df(cli.val_csv)
test_df  = load_df(cli.test_csv)

# -------------------- TAPT (MLM) --------------------
tapt_dir = os.path.join(RUN_DIR, "tapt_mlm")
os.makedirs(tapt_dir, exist_ok=True)

tok_mlm = AutoTokenizer.from_pretrained(cli.base_model, use_fast=True)
def tok_text(batch): return tok_mlm(batch["text_bert"], truncation=True, max_length=cli.tapt_max_len)

# TAPT veri kaynağı: Train (+ dilersen Val'i de birleştirebilirsin)
tapt_texts = pd.concat([train_df[["text_bert"]]], axis=0, ignore_index=True)
tapt_ds = Dataset.from_pandas(tapt_texts, preserve_index=False).map(tok_text, batched=True, remove_columns=["text_bert"])
mlm_collator = DataCollatorForLanguageModeling(tokenizer=tok_mlm, mlm_probability=cli.tapt_mlm_prob)

mlm_model = AutoModelForMaskedLM.from_pretrained(cli.base_model)

args_mlm = TrainingArguments(
    output_dir=tapt_dir,
    per_device_train_batch_size=cli.tapt_batch,
    gradient_accumulation_steps=cli.tapt_grad_accum,
    learning_rate=cli.tapt_lr,
    num_train_epochs=cli.tapt_epochs,
    weight_decay=cli.tapt_wd,
    warmup_ratio=cli.tapt_warmup,
    logging_steps=100,
    report_to="none",
    save_strategy="no",
    seed=SEED
)
mlm_trainer = Trainer(
    model=mlm_model,
    args=args_mlm,
    train_dataset=tapt_ds,
    data_collator=mlm_collator,
    tokenizer=tok_mlm
)
print("[TAPT] Starting MLM pretraining…")
mlm_trainer.train()
mlm_trainer.save_model(tapt_dir)
print("[TAPT] Done.")

# -------------------- Supervised fine-tune (Train + Val selection) --------------------
ft_sel_dir = os.path.join(RUN_DIR, "finetune_valselect")
os.makedirs(ft_sel_dir, exist_ok=True)

tok_cls = AutoTokenizer.from_pretrained(tapt_dir, use_fast=True)
collator = DataCollatorWithPadding(tokenizer=tok_cls)

train_ds = build_cls_dataset(train_df, tok_cls, cli.ft_max_len)
val_ds   = build_cls_dataset(val_df,   tok_cls, cli.ft_max_len)
test_ds  = build_cls_dataset(test_df,  tok_cls, cli.ft_max_len)  # sadece finalde kullanılacak

cls_model = AutoModelForSequenceClassification.from_pretrained(
    tapt_dir, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)

args_ft = TrainingArguments(
    output_dir=ft_sel_dir,
    per_device_train_batch_size=cli.ft_batch,
    per_device_eval_batch_size=cli.ft_batch,
    gradient_accumulation_steps=cli.ft_grad_accum,
    learning_rate=cli.ft_lr,
    num_train_epochs=cli.ft_epochs,
    weight_decay=cli.ft_wd,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
    seed=SEED
)

trainer_sel = Trainer(
    model=cls_model,
    args=args_ft,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok_cls,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cli.ft_patience)]
)

print("[FT] Train on Train, select on Val…")
ckpt = latest_ckpt(ft_sel_dir)
trainer_sel.train(resume_from_checkpoint=ckpt if ckpt else None)

# Val performansını kaydet
val_eval = trainer_sel.evaluate()
with open(os.path.join(ft_sel_dir, "val_metrics.json"), "w") as f:
    json.dump({k: float(v) for k, v in val_eval.items()}, f, indent=2)

# -------------------- Refit on Train+Val (eval kapalı) --------------------
refit_dir = os.path.join(RUN_DIR, "refit_trainval")
os.makedirs(refit_dir, exist_ok=True)

trainval_df = pd.concat([train_df, val_df], ignore_index=True)
trainval_ds = build_cls_dataset(trainval_df, tok_cls, cli.ft_max_len)

refit_model = AutoModelForSequenceClassification.from_pretrained(
    tapt_dir, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)

args_refit = TrainingArguments(
    output_dir=refit_dir,
    per_device_train_batch_size=cli.ft_batch,
    per_device_eval_batch_size=cli.ft_batch,
    gradient_accumulation_steps=cli.ft_grad_accum,
    learning_rate=cli.ft_lr,
    num_train_epochs=cli.ft_epochs,
    weight_decay=cli.ft_wd,
    eval_strategy="no",                # test sızıntısını önler
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    logging_steps=100,
    report_to="none",
    seed=SEED
)

trainer_refit = Trainer(
    model=refit_model,
    args=args_refit,
    train_dataset=trainval_ds,
    tokenizer=tok_cls,
    data_collator=collator
)

print("[REFIT] Train on Train+Val (no eval)…")
ckpt = latest_ckpt(refit_dir)
trainer_refit.train(resume_from_checkpoint=ckpt if ckpt else None)

# -------------------- Final single-shot Test --------------------
print("[TEST] Evaluating on Test (single shot)…")
test_eval = trainer_refit.evaluate(test_ds)
pred = trainer_refit.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

# Rapor/tablolar
rep = classification_report(
    y_true, y_pred, target_names=[ID2LABEL[i] for i in [0,1,2]],
    output_dict=True, digits=4
)
rep_df = pd.DataFrame(rep).transpose()
rep_df.to_csv(os.path.join(refit_dir, "bert_test_report.csv"))

acc = accuracy_score(y_true, y_pred)
p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

overview = {
    "accuracy": acc,
    "macro_precision": p_m, "macro_recall": r_m, "macro_f1": f_m, "weighted_f1": f_w,
    "tapt_lr": cli.tapt_lr, "tapt_epochs": cli.tapt_epochs, "tapt_max_len": cli.tapt_max_len,
    "ft_lr": cli.ft_lr, "ft_epochs": cli.ft_epochs, "ft_max_len": cli.ft_max_len,
    "ft_batch": cli.ft_batch, "ft_grad_accum": cli.ft_grad_accum
}
pd.DataFrame([overview]).to_csv(os.path.join(refit_dir, "bert_overview_metrics_test.csv"), index=False)

# Sınıf bazlı tablo
cls_tbl = rep_df.loc[["negative","neutral","positive"], ["precision","recall","f1-score","support"]].copy()
cls_tbl.columns = ["precision","recall","f1","support"]
cls_tbl.to_csv(os.path.join(refit_dir, "TABLE_classwise_metrics.csv"), index=True)

# CM CSV + PNG
save_cm_png_csv(
    y_true, y_pred,
    os.path.join(refit_dir, "bert_confusion_matrix_test.csv"),
    os.path.join(refit_dir, "FIG_confusion_matrix_test.png")
)

# Test tahminleri
pd.DataFrame({
    "text_bert": test_df["text_bert"],
    "sentiment": test_df["sentiment"],
    "pred_id": y_pred,
    "pred_label": [ID2LABEL[i] for i in y_pred]
}).to_csv(os.path.join(refit_dir, "bert_test_predictions.csv"), index=False)

# Çalışma özeti
summary = {
    "paths": {
        "run_dir": RUN_DIR,
        "tapt_dir": tapt_dir,
        "finetune_valselect_dir": ft_sel_dir,
        "refit_dir": refit_dir
    },
    "val_metrics": {k: float(v) for k, v in val_eval.items()},
    "test_overview": overview
}
with open(os.path.join(RUN_DIR, "RUN_SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Phase 2.4 complete ===")
print(f"- RUN_DIR: {RUN_DIR}")
print(f"- Test overview CSV: {os.path.join(refit_dir, 'bert_overview_metrics_test.csv')}")
print(f"- Class-wise CSV:    {os.path.join(refit_dir, 'TABLE_classwise_metrics.csv')}")
print(f"- Report CSV:        {os.path.join(refit_dir, 'bert_test_report.csv')}")
print(f"- CM PNG:            {os.path.join(refit_dir, 'FIG_confusion_matrix_test.png')}")
