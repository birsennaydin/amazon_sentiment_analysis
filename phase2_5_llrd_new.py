# Phase 2.5 — LLRD + Label Smoothing (+ optional Temperature Scaling) + Multi-Seed
# Pipeline: Train→Val selection (ES) → Refit on Train+Val (no eval) → Single-shot Test
# Outputs: results/phase2/phase2_5_new_test/<timestamp>/seed_<SEED>/{finetune_valselect, refit_trainval}/...

import os, re, json, random, gc, platform
from datetime import datetime
from glob import glob

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Apple MPS OOM'u azaltır

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from datasets import Dataset
import matplotlib.pyplot as plt

from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
)
from torch.optim import AdamW

# ---------- tiny console helpers ----------
def banner(msg):
    print("\n" + "="*20 + f" {msg} " + "="*20)

def print_class_dist(name, df):
    counts = df["sentiment"].astype(str).str.lower().value_counts()
    total = len(df)
    parts = [f"{k}:{counts.get(k,0)}" for k in ["negative","neutral","positive"]]
    print(f"{name}: {total} rows  |  " + "  ".join(parts))

# -------------------- CLI --------------------
import argparse
ap = argparse.ArgumentParser(description="Phase 2.5 — LLRD + Seeds + Label Smoothing (val→refit→test)")
ap.add_argument("--train_csv", default="data/train.csv")
ap.add_argument("--val_csv",   default="data/val.csv")
ap.add_argument("--test_csv",  default="data/test.csv")
ap.add_argument("--base_model", default="bert-base-uncased")

# Fine-tune (Val selection)
ap.add_argument("--ft_max_len", type=int, default=128)
ap.add_argument("--ft_epochs", type=int, default=4)
ap.add_argument("--ft_batch", type=int, default=4)
ap.add_argument("--ft_grad_accum", type=int, default=8)
ap.add_argument("--ft_warmup", type=float, default=0.10)
ap.add_argument("--label_smoothing", type=float, default=0.05)
ap.add_argument("--ft_patience", type=int, default=2)

# LLRD
ap.add_argument("--llrd_base_lr", type=float, default=1.5e-5)
ap.add_argument("--llrd_decay", type=float, default=0.95)
ap.add_argument("--llrd_head_lr", type=float, default=3e-5)
ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--max_grad_norm", type=float, default=1.0)

# Partial FT (memory-friendly)
ap.add_argument("--unfreeze_last_n", type=int, default=1)

# Temperature scaling (metrics-only)
ap.add_argument("--temp_scale", type=float, default=1.0)

# Seeds
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])

args = ap.parse_args()

# -------------------- Consts & paths --------------------
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ROOT = "results/phase2/phase2_5_neww_test"
os.makedirs(ROOT, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(ROOT, STAMP)
os.makedirs(RUN_DIR, exist_ok=True)

banner("PHASE 2.5 — LLRD + Label Smoothing + Multi-Seed")
print("Timestamp:", datetime.now().isoformat(timespec="seconds"))
print("Python:", platform.python_version(), "| Torch:", torch.__version__)
print("Model:", args.base_model, "| Seeds:", args.seeds)
print("Run dir:", RUN_DIR)

# -------------------- Utils --------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "text_bert" in df.columns and "sentiment" in df.columns, "CSV must have text_bert & sentiment"
    df = df.dropna(subset=["text_bert","sentiment"]).copy()
    df["label"] = df["sentiment"].astype(str).str.lower().map(LABEL2ID)
    df = df[df["label"].isin([0,1,2])].reset_index(drop=True)
    return df

def build_cls_dataset(df: pd.DataFrame, tokenizer, max_len: int):
    def tok(b): return tokenizer(b["text_bert"], truncation=True, max_length=max_len)
    ds = Dataset.from_pandas(df[["text_bert","label"]].rename(columns={"label":"labels"}), preserve_index=False)
    return ds.map(tok, batched=True, remove_columns=["text_bert"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if args.temp_scale != 1.0:
        logits = logits / args.temp_scale
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

def save_cm_png_csv(y_true, y_pred, out_csv, out_png, title="Confusion Matrix — BERT (Phase 2.5)"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_df = pd.DataFrame(cm, index=[ID2LABEL[i] for i in [0,1,2]],
                         columns=[ID2LABEL[i] for i in [0,1,2]])
    cm_df.to_csv(out_csv)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(ticks=range(3), labels=list(cm_df.columns))
    plt.yticks(ticks=range(3), labels=list(cm_df.index))
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

def build_llrd_param_groups(model: BertForSequenceClassification,
                            llrd_base_lr: float,
                            llrd_decay: float,
                            llrd_head_lr: float,
                            weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight"]
    groups = []
    num_layers = len(model.bert.encoder.layer)

    # Encoder layers (lower -> higher)
    for idx in range(num_layers):
        lr = llrd_base_lr * (llrd_decay ** (num_layers - 1 - idx))
        layer = model.bert.encoder.layer[idx]
        decay, nodecay = [], []
        for n, p in layer.named_parameters():
            (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
        if decay:   groups.append({"params": decay,   "lr": lr, "weight_decay": weight_decay})
        if nodecay: groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})

    # Embeddings
    emb_decay, emb_nodecay = [], []
    for n, p in model.bert.embeddings.named_parameters():
        (emb_nodecay if any(nd in n for nd in no_decay) else emb_decay).append(p)
    emb_lr = llrd_base_lr * (llrd_decay ** num_layers)
    if emb_decay:   groups.append({"params": emb_decay,   "lr": emb_lr, "weight_decay": weight_decay})
    if emb_nodecay: groups.append({"params": emb_nodecay, "lr": emb_lr, "weight_decay": 0.0})

    # Pooler (if exists)
    if getattr(model.bert, "pooler", None) is not None:
        pool_decay, pool_nodecay = [], []
        for n, p in model.bert.pooler.named_parameters():
            (pool_nodecay if any(nd in n for nd in no_decay) else pool_decay).append(p)
        if pool_decay:   groups.append({"params": pool_decay,   "lr": llrd_base_lr, "weight_decay": weight_decay})
        if pool_nodecay: groups.append({"params": pool_nodecay, "lr": llrd_base_lr, "weight_decay": 0.0})

    # Classifier head (higher LR)
    head_decay, head_nodecay = [], []
    for n, p in model.classifier.named_parameters():
        (head_nodecay if any(nd in n for nd in no_decay) else head_decay).append(p)
    if head_decay:   groups.append({"params": head_decay,   "lr": llrd_head_lr, "weight_decay": weight_decay})
    if head_nodecay: groups.append({"params": head_nodecay, "lr": llrd_head_lr, "weight_decay": 0.0})

    return groups

def apply_partial_finetune(model: BertForSequenceClassification, unfreeze_last_n: int):
    # Freeze all encoder
    for p in model.bert.parameters():
        p.requires_grad = False
    # Unfreeze last N blocks
    if unfreeze_last_n > 0:
        for layer in model.bert.encoder.layer[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
    # Pooler (if exists)
    if getattr(model.bert, "pooler", None) is not None:
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
    # Always train classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

# Custom Trainer to inject LLRD optimizer
class LLRDTrainer(Trainer):
    def __init__(self, *args, llrd_param_groups=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llrd_param_groups = llrd_param_groups

    def create_optimizer(self):
        if self.optimizer is None:
            if self.llrd_param_groups is None:
                self.optimizer = AdamW(self.model.parameters(),
                                       lr=self.args.learning_rate,
                                       betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=self.args.weight_decay)
            else:
                self.optimizer = AdamW(self.llrd_param_groups,
                                       betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

# -------------------- Data --------------------
train_df = load_df(args.train_csv)
val_df   = load_df(args.val_csv)
test_df  = load_df(args.test_csv)

banner("DATA SUMMARY")
print_class_dist("TRAIN", train_df)
print_class_dist("VAL  ", val_df)
print_class_dist("TEST ", test_df)

# -------------------- Run per-seed --------------------
all_seed_rows = []
for seed in args.seeds:
    set_seed(seed)
    SEED_DIR = os.path.join(RUN_DIR, f"seed_{seed}")
    os.makedirs(SEED_DIR, exist_ok=True)

    banner(f"SEED {seed} — START")
    print("Output dir:", SEED_DIR)

    # Tokeniser & collator
    tokenizer = BertTokenizerFast.from_pretrained(args.base_model, use_fast=True)
    collator  = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = build_cls_dataset(train_df, tokenizer, args.ft_max_len)
    val_ds   = build_cls_dataset(val_df,   tokenizer, args.ft_max_len)
    test_ds  = build_cls_dataset(test_df,  tokenizer, args.ft_max_len)

    # --------- Model (with partial FT) ----------
    model = BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.gradient_checkpointing_enable()
    apply_partial_finetune(model, args.unfreeze_last_n)

    # LLRD param groups
    llrd_groups = build_llrd_param_groups(
        model, args.llrd_base_lr, args.llrd_decay, args.llrd_head_lr, args.weight_decay
    )

    # --------- Train on Train, select on Val ---------
    FT_DIR = os.path.join(SEED_DIR, "finetune_valselect")
    os.makedirs(FT_DIR, exist_ok=True)

    ft_args = TrainingArguments(
        output_dir=FT_DIR,
        per_device_train_batch_size=args.ft_batch,
        per_device_eval_batch_size=args.ft_batch,
        gradient_accumulation_steps=args.ft_grad_accum,
        learning_rate=args.llrd_head_lr,  # gerçek LR param grup’larda
        weight_decay=args.weight_decay,
        warmup_ratio=args.ft_warmup,
        num_train_epochs=args.ft_epochs,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        label_smoothing_factor=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        group_by_length=True,
        eval_accumulation_steps=32
    )

    trainer_sel = LLRDTrainer(
        model=model,
        args=ft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.ft_patience)],
        llrd_param_groups=llrd_groups,
    )

    print(f"[FT][seed={seed}] Train on Train, select on Val…")
    ckpt = latest_ckpt(FT_DIR)
    trainer_sel.train(resume_from_checkpoint=ckpt if ckpt else None)

    val_eval = trainer_sel.evaluate()
    print(f"[VAL][seed={seed}] accuracy={val_eval.get('eval_accuracy',0):.4f} "
          f"macro_f1={val_eval.get('eval_macro_f1',0):.4f}")
    with open(os.path.join(FT_DIR, "val_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in val_eval.items()}, f, indent=2)

    # --------- Refit on Train+Val (no eval) ---------
    REFIT_DIR = os.path.join(SEED_DIR, "refit_trainval")
    os.makedirs(REFIT_DIR, exist_ok=True)

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    trainval_ds = build_cls_dataset(trainval_df, tokenizer, args.ft_max_len)

    refit_model = BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    refit_model.gradient_checkpointing_enable()
    apply_partial_finetune(refit_model, args.unfreeze_last_n)

    refit_groups = build_llrd_param_groups(
        refit_model, args.llrd_base_lr, args.llrd_decay, args.llrd_head_lr, args.weight_decay
    )

    refit_args = TrainingArguments(
        output_dir=REFIT_DIR,
        per_device_train_batch_size=args.ft_batch,
        per_device_eval_batch_size=args.ft_batch,
        gradient_accumulation_steps=args.ft_grad_accum,
        learning_rate=args.llrd_head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.ft_warmup,
        num_train_epochs=args.ft_epochs,
        lr_scheduler_type="linear",
        eval_strategy="no",            # test sızıntısı yok
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        label_smoothing_factor=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        group_by_length=True
    )

    trainer_refit = LLRDTrainer(
        model=refit_model,
        args=refit_args,
        train_dataset=trainval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        llrd_param_groups=refit_groups
    )

    print(f"[REFIT][seed={seed}] Train on Train+Val (no eval)…")
    ckpt = latest_ckpt(REFIT_DIR)
    trainer_refit.train(resume_from_checkpoint=ckpt if ckpt else None)
    print(f"[REFIT][seed={seed}] done.")

    # --------- Single-shot Test ---------
    print(f"[TEST][seed={seed}] Evaluating on Test once…")
    pred = trainer_refit.predict(test_ds)
    y_true = pred.label_ids
    logits = pred.predictions
    if args.temp_scale != 1.0:
        logits = logits / args.temp_scale
    y_pred = np.argmax(logits, axis=1)

    rep = classification_report(
        y_true, y_pred, target_names=[ID2LABEL[i] for i in [0,1,2]],
        output_dict=True, digits=4
    )
    rep_df = pd.DataFrame(rep).transpose()
    rep_df.to_csv(os.path.join(REFIT_DIR, "bert_test_report.csv"))

    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    overview = {
        "seed": seed,
        "accuracy": acc,
        "macro_precision": p_m, "macro_recall": r_m, "macro_f1": f_m, "weighted_f1": f_w,
        "llrd_base_lr": args.llrd_base_lr, "llrd_decay": args.llrd_decay, "llrd_head_lr": args.llrd_head_lr,
        "ft_max_len": args.ft_max_len, "ft_epochs": args.ft_epochs, "ft_batch": args.ft_batch,
        "ft_grad_accum": args.ft_grad_accum, "label_smoothing": args.label_smoothing,
        "temp_scale": args.temp_scale, "unfreeze_last_n": args.unfreeze_last_n
    }
    pd.DataFrame([overview]).to_csv(os.path.join(REFIT_DIR, "bert_overview_metrics_test.csv"), index=False)

    # Sınıf-bazlı tablo
    cls_tbl = rep_df.loc[["negative","neutral","positive"], ["precision","recall","f1-score","support"]].copy()
    cls_tbl.columns = ["precision","recall","f1","support"]
    cls_tbl.to_csv(os.path.join(REFIT_DIR, "TABLE_classwise_metrics.csv"), index=True)

    # CM
    save_cm_png_csv(
        y_true, y_pred,
        os.path.join(REFIT_DIR, "bert_confusion_matrix_test.csv"),
        os.path.join(REFIT_DIR, "FIG_confusion_matrix_test.png"),
        title="Confusion Matrix — BERT (Phase 2.5, Test)"
    )

    # Test tahminleri
    pd.DataFrame({
        "text_bert": test_df["text_bert"],
        "sentiment": test_df["sentiment"],
        "pred_id": y_pred,
        "pred_label": [ID2LABEL[i] for i in y_pred]
    }).to_csv(os.path.join(REFIT_DIR, "bert_test_predictions.csv"), index=False)

    # Val metriklerini üst düzeye kaydet
    with open(os.path.join(SEED_DIR, "val_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in val_eval.items()}, f, indent=2)

    # Konsola kısa sınıf özeti
    print(f"[TEST][seed={seed}] accuracy={acc:.4f} macro_f1={f_m:.4f} (weighted_f1={f_w:.4f})")
    for lbl in ["negative","neutral","positive"]:
        row = rep_df.loc[lbl]
        print(f"  - {lbl:<8} P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1-score']:.3f} N={int(row['support'])}")

    # Özet satırı
    all_seed_rows.append([seed, acc, p_m, r_m, f_m, f_w])

    banner(f"SEED {seed} — DONE")

    # Temizlik
    del trainer_sel, trainer_refit, model, refit_model
    gc.collect()
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except Exception: pass

# -------------------- Aggregate across seeds --------------------
if all_seed_rows:
    agg = pd.DataFrame(all_seed_rows, columns=["seed","accuracy","macro_precision","macro_recall","macro_f1","weighted_f1"])
    agg.to_csv(os.path.join(RUN_DIR, "by_seed_test_metrics.csv"), index=False)
    mean = agg.mean(numeric_only=True); std = agg.std(numeric_only=True)
    pd.DataFrame([{
        "accuracy_mean": float(mean["accuracy"]), "accuracy_std": float(std["accuracy"]),
        "macro_precision_mean": float(mean["macro_precision"]), "macro_precision_std": float(std["macro_precision"]),
        "macro_recall_mean": float(mean["macro_recall"]), "macro_recall_std": float(std["macro_recall"]),
        "macro_f1_mean": float(mean["macro_f1"]), "macro_f1_std": float(std["macro_f1"]),
        "weighted_f1_mean": float(mean["weighted_f1"]), "weighted_f1_std": float(std["weighted_f1"])
    }]).to_csv(os.path.join(RUN_DIR, "summary_mean_std.csv"), index=False)

# -------------------- Run summary --------------------
banner("FILES WRITTEN (top-level)")
for name in sorted(os.listdir(RUN_DIR)):
    p = os.path.join(RUN_DIR, name)
    if os.path.isdir(p):
        print("DIR :", name)
        for sub in sorted(os.listdir(p))[:5]:
            print("   -", os.path.join(name, sub))
    else:
        print("FILE:", name)

summary = {
    "paths": {"run_dir": RUN_DIR},
    "seeds": args.seeds
}
with open(os.path.join(RUN_DIR, "RUN_SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Phase 2.5 complete ===")
print(f"- RUN_DIR: {RUN_DIR}")
print(f"- By-seed metrics: {os.path.join(RUN_DIR, 'by_seed_test_metrics.csv')}")
print(f"- Mean±Std:       {os.path.join(RUN_DIR, 'summary_mean_std.csv')}")
