# Phase 2.5 — LLRD + Multi-Seed + Label Smoothing (+ optional Temperature Scaling)
# Outputs under: results/phase2/phase2_5/<timestamp or --resume_dir>/seed_<SEED>/
# M1 (8GB) friendly: small batch + grad accumulation + gradient checkpointing + memory cleanup.

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import gc, torch

from glob import glob
from safetensors.torch import load_file as safe_load_file

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

from datasets import load_dataset, DatasetDict

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from torch.optim import AdamW


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
    for c in columns:
        if c.lower() not in {"label", "labels", "sentiment", "target", "y"}:
            return c
    raise ValueError("No suitable text column found. Set --text_column.")

def build_label_mapping(series: pd.Series) -> Dict[str, int]:
    if pd.api.types.is_integer_dtype(series) and series.min() >= 0 and series.nunique() <= 20:
        classes = sorted(series.unique().tolist())
        return {str(c): int(c) for c in classes}
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

def tokenize_for_cls(tokenizer, text_col, max_len):
    def fn(batch: Dict[str, Any]):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_len,
            padding=False,
            return_token_type_ids=False,
        )
    return fn


# ---------------------------
# LLRD param groups
# ---------------------------

def build_llrd_param_groups(model: BertForSequenceClassification,
                            llrd_base_lr: float,
                            llrd_decay: float,
                            llrd_head_lr: float,
                            weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = []

    num_layers = len(model.bert.encoder.layer)
    for layer_idx in range(num_layers):
        lr = llrd_base_lr * (llrd_decay ** (num_layers - 1 - layer_idx))
        layer = model.bert.encoder.layer[layer_idx]
        decay_params, nodecay_params = [], []
        for n, p in layer.named_parameters():
            (nodecay_params if any(nd in n for nd in no_decay) else decay_params).append(p)
        if decay_params:
            param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
        if nodecay_params:
            param_groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})

    emb_decay, emb_nodecay = [], []
    for n, p in model.bert.embeddings.named_parameters():
        (emb_nodecay if any(nd in n for nd in no_decay) else emb_decay).append(p)
    emb_lr = llrd_base_lr * (llrd_decay ** num_layers)
    if emb_decay:
        param_groups.append({"params": emb_decay, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_nodecay:
        param_groups.append({"params": emb_nodecay, "lr": emb_lr, "weight_decay": 0.0})

    if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
        pool_decay, pool_nodecay = [], []
        for n, p in model.bert.pooler.named_parameters():
            (pool_nodecay if any(nd in n for nd in no_decay) else pool_decay).append(p)
        top_lr = llrd_base_lr
        if pool_decay:
            param_groups.append({"params": pool_decay, "lr": top_lr, "weight_decay": weight_decay})
        if pool_nodecay:
            param_groups.append({"params": pool_nodecay, "lr": top_lr, "weight_decay": 0.0})

    head_decay, head_nodecay = [], []
    for n, p in model.classifier.named_parameters():
        (head_nodecay if any(nd in n for nd in no_decay) else head_decay).append(p)
    if head_decay:
        param_groups.append({"params": head_decay, "lr": llrd_head_lr, "weight_decay": weight_decay})
    if head_nodecay:
        param_groups.append({"params": head_nodecay, "lr": llrd_head_lr, "weight_decay": 0.0})

    return param_groups


# ---------------------------
# Partial FT helper (optional, memory friendly)
# ---------------------------

def apply_partial_finetune(model: BertForSequenceClassification, unfreeze_last_n: int):
    for p in model.bert.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        for layer in model.bert.encoder.layer[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
    if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    print(f"[FT] Partial FT: last {unfreeze_last_n} encoder block(s) + classifier are trainable.")


# ---------------------------
# Custom Trainer to inject LLRD optimizer
# ---------------------------

class LLRDTrainer(Trainer):
    def __init__(self, *args, llrd_param_groups=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llrd_param_groups = llrd_param_groups

    def create_optimizer(self):
        if self.optimizer is None:
            if self.llrd_param_groups is None:
                self.optimizer = AdamW(self.model.parameters(),
                                       lr=self.args.learning_rate,
                                       betas=(0.9, 0.999),
                                       eps=1e-8,
                                       weight_decay=self.args.weight_decay)
            else:
                self.optimizer = AdamW(self.llrd_param_groups,
                                       betas=(0.9, 0.999),
                                       eps=1e-8)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)


# ---------------------------
# Metrics (+ optional temperature scaling)
# ---------------------------

def softmax_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits if T == 1.0 else logits / T

def compute_metrics_builder(temp: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.array(logits); labels = np.array(labels)
        preds = np.argmax(softmax_temperature(logits, temp), axis=1)
        acc = (preds == labels).mean()
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": float(acc), "f1": float(f1), "precision": float(prec), "recall": float(rec)}
    return compute_metrics


# ---------------------------
# Robust checkpoint picker
# ---------------------------

def find_latest_healthy_checkpoint(seed_out: str) -> str | None:
    """
    seed_out içindeki checkpoint-* klasörlerini (step'e göre) tersten dener.
    model.safetensors yüklenebiliyorsa sağlam kabul eder.
    """
    if not os.path.isdir(seed_out):
        return None
    ckpts = sorted(
        [p for p in glob(os.path.join(seed_out, "checkpoint-*")) if os.path.isdir(p)],
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
        reverse=True,
    )
    for ck in ckpts:
        wt = os.path.join(ck, "model.safetensors")
        if not os.path.isfile(wt):
            continue
        try:
            _ = safe_load_file(wt, device="cpu")  # header/deser testi; bozuksa patlar
            return ck
        except Exception as e:
            print(f"[WARN] Checkpoint seems corrupted, skipping: {ck} ({e})")
    return None


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2.5 — LLRD + Multi-Seed + Label Smoothing (+ Temp Scaling)")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/val.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="sentiment")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")

    # Fine-tune params
    parser.add_argument("--ft_epochs", type=int, default=4)
    parser.add_argument("--ft_max_len", type=int, default=128)
    parser.add_argument("--ft_batch", type=int, default=4)
    parser.add_argument("--ft_grad_accum", type=int, default=8)
    parser.add_argument("--ft_warmup", type=float, default=0.10)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    # LLRD params
    parser.add_argument("--llrd_base_lr", type=float, default=1.5e-5)
    parser.add_argument("--llrd_decay", type=float, default=0.95)
    parser.add_argument("--llrd_head_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Temperature scaling (metrics only)
    parser.add_argument("--temp_scale", type=float, default=1.0)

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])

    # Partial FT
    parser.add_argument("--unfreeze_last_n", type=int, default=1)

    # Output root or resume into an existing timestamp dir
    parser.add_argument("--root_out", type=str, default="results/phase2/phase2_5")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Var olan timestamp klasörü. Verilirse yeni timestamp oluşturulmaz.")
    parser.add_argument("--skip_completed", action="store_true",
                        help="seed_X klasöründe test_metrics.csv varsa seedi atla ve metriklerini özetle.")

    args = parser.parse_args()

    if args.resume_dir:
        ROOT_TS = args.resume_dir
        if not os.path.isdir(ROOT_TS):
            raise ValueError(f"--resume_dir not found: {ROOT_TS}")
        print(f"[INFO] Resuming into existing dir: {ROOT_TS}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ROOT_TS = os.path.join(args.root_out, timestamp)
        ensure_dir(ROOT_TS)
        print(f"[INFO] Output root: {ROOT_TS}")

    print(f"[INFO] Device: {get_device_str()}")
    print(f"[INFO] Seeds: {args.seeds}")

    # Load CSVs
    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)
    df_test  = pd.read_csv(args.test_csv)

    text_col = args.text_column or detect_text_column(df_train.columns.tolist())
    if text_col not in df_train.columns:
        raise ValueError(f"Text column '{text_col}' not in train CSV.")
    if args.label_column not in df_train.columns:
        raise ValueError(f"Label column '{args.label_column}' not in train CSV.")

    label_map = build_label_mapping(df_train[args.label_column])
    inv_map = {v: k for k, v in label_map.items()}  # <<<<<< ADDED (id -> name)

    dataset = DatasetDict({
        "train": load_dataset("csv", data_files={"train": args.train_csv})["train"],
        "val":   load_dataset("csv", data_files={"val": args.val_csv})["val"],
        "test":  load_dataset("csv", data_files={"test": args.test_csv})["test"],
    })

    def mapped(ds_split, df_split):
        cols_to_remove = [c for c in ds_split.column_names if c != text_col]
        ds = ds_split.remove_columns(cols_to_remove)
        labels = apply_label_mapping(df_split[args.label_column], label_map).tolist()
        return ds.add_column("label", labels)

    dataset["train"] = mapped(dataset["train"], df_train)
    dataset["val"]   = mapped(dataset["val"],   df_val)
    dataset["test"]  = mapped(dataset["test"],  df_test)

    with open(os.path.join(ROOT_TS, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    tokenizer = BertTokenizerFast.from_pretrained(args.base_model)
    ds_tok_train = dataset["train"].map(tokenize_for_cls(tokenizer, text_col, args.ft_max_len), batched=True)
    ds_tok_val   = dataset["val"].map(tokenize_for_cls(tokenizer, text_col, args.ft_max_len), batched=True)
    ds_tok_test  = dataset["test"].map(tokenize_for_cls(tokenizer, text_col, args.ft_max_len), batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # <<<<<< ADDED: Trainer'ın beklediği isim
    ds_tok_train = ds_tok_train.rename_column("label", "labels")
    ds_tok_val   = ds_tok_val.rename_column("label", "labels")
    ds_tok_test  = ds_tok_test.rename_column("label", "labels")

    all_seed_metrics = []

    for seed in args.seeds:
        set_seed(seed)
        SEED_OUT = os.path.join(ROOT_TS, f"seed_{seed}")
        ensure_dir(SEED_OUT)

        # Skip if already completed
        tm_path = os.path.join(SEED_OUT, "test_metrics.csv")
        if args.skip_completed and os.path.isfile(tm_path):
            print(f"[SKIP][seed={seed}] test_metrics.csv bulundu, bu seedi atlıyorum.")
            try:
                dfm = pd.read_csv(tm_path)
                all_seed_metrics.append([
                    seed,
                    float(dfm["accuracy"].iloc[0]),
                    float(dfm["macro_precision"].iloc[0]),
                    float(dfm["macro_recall"].iloc[0]),
                    float(dfm["macro_f1"].iloc[0]),
                ])
            except Exception:
                pass
            continue

        # <<<<<< CHANGED: id2label/label2id ver
        model = BertForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=len(label_map),
            id2label=inv_map,
            label2id=label_map
        )
        model.gradient_checkpointing_enable()
        apply_partial_finetune(model, unfreeze_last_n=args.unfreeze_last_n)

        llrd_groups = build_llrd_param_groups(
            model=model,
            llrd_base_lr=args.llrd_base_lr,
            llrd_decay=args.llrd_decay,
            llrd_head_lr=args.llrd_head_lr,
            weight_decay=args.weight_decay,
        )

        ft_args = TrainingArguments(
            output_dir=SEED_OUT,
            overwrite_output_dir=False,      # resume için False
            num_train_epochs=args.ft_epochs,
            per_device_train_batch_size=args.ft_batch,
            gradient_accumulation_steps=args.ft_grad_accum,
            learning_rate=args.llrd_head_lr, # placeholder; gerçek lr param gruplarda
            weight_decay=args.weight_decay,
            warmup_ratio=args.ft_warmup,
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,              # 2 checkpoint tut (bozulma anında geri düşebilmek için)
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            group_by_length=True,
            eval_accumulation_steps=32,
            max_grad_norm=args.max_grad_norm,
            label_smoothing_factor=args.label_smoothing,
            seed=seed,
        )

        compute_metrics = compute_metrics_builder(temp=args.temp_scale)

        trainer = LLRDTrainer(
            model=model,
            args=ft_args,
            train_dataset=ds_tok_train,
            eval_dataset=ds_tok_val,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            llrd_param_groups=llrd_groups,
        )

        # ---- Robust resume logic ----
        resume_ckpt = find_latest_healthy_checkpoint(SEED_OUT)
        if resume_ckpt is not None:
            print(f"[FT][seed={seed}] Resuming from checkpoint: {resume_ckpt}")
        else:
            print(f"[FT][seed={seed}] Fresh start (no healthy checkpoint found).")
        # -----------------------------

        print(f"[FT][seed={seed}] Training started...")
        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(SEED_OUT)
        tokenizer.save_pretrained(SEED_OUT)
        print(f"[FT][seed={seed}] Saved best model to {SEED_OUT}")

        # ---- Test evaluation
        print(f"[EVAL][seed={seed}] Running test evaluation...")
        preds_raw = trainer.predict(ds_tok_test)

        # <<<<<< CHANGED: ground-truth'u Trainer'dan al
        y_true = preds_raw.label_ids
        y_pred = np.argmax(softmax_temperature(preds_raw.predictions, args.temp_scale), axis=1)

        target_names = [inv_map[i] for i in range(len(inv_map))]

        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(SEED_OUT, "classification_report.csv"))

        acc = float((y_true == y_pred).mean())
        macro_prec = float(report_df.loc["macro avg", "precision"])
        macro_rec  = float(report_df.loc["macro avg", "recall"])
        macro_f1   = float(report_df.loc["macro avg", "f1-score"])

        pd.DataFrame([{
            "seed": seed,
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
        }]).to_csv(os.path.join(SEED_OUT, "test_metrics.csv"), index=False)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
        plot_confusion_matrix(cm, target_names, os.path.join(SEED_OUT, "confusion_matrix.png"))

        pd.DataFrame({
            "id": np.arange(len(y_true)),
            "true_label": [inv_map[int(i)] for i in y_true],
            "pred_label": [inv_map[int(i)] for i in y_pred],
        }).to_csv(os.path.join(SEED_OUT, "predictions.csv"), index=False)

        all_seed_metrics.append([seed, acc, macro_prec, macro_rec, macro_f1])

        del trainer, model
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    # ---- Aggregate across seeds
    if all_seed_metrics:
        agg = pd.DataFrame(all_seed_metrics, columns=["seed", "accuracy", "macro_precision", "macro_recall", "macro_f1"])
        agg_mean = agg.mean(numeric_only=True); agg_std = agg.std(numeric_only=True)
        pd.DataFrame([{
            "accuracy_mean": round(float(agg_mean["accuracy"]), 4),
            "accuracy_std":  round(float(agg_std["accuracy"]), 4),
            "macro_precision_mean": round(float(agg_mean["macro_precision"]), 4),
            "macro_precision_std":  round(float(agg_std["macro_precision"]), 4),
            "macro_recall_mean": round(float(agg_mean["macro_recall"]), 4),
            "macro_recall_std":  round(float(agg_std["macro_recall"]), 4),
            "macro_f1_mean": round(float(agg_mean["macro_f1"]), 4),
            "macro_f1_std":  round(float(agg_std["macro_f1"]), 4),
        }]).to_csv(os.path.join(ROOT_TS, "summary_mean_std.csv"), index=False)
        agg.to_csv(os.path.join(ROOT_TS, "by_seed_test_metrics.csv"), index=False)
    else:
        print("[WARN] No metrics collected (all seeds may have been skipped and reading failed).")

    print(f"[DONE] Seed-wise and summary metrics are saved under: {ROOT_TS}")


if __name__ == "__main__":
    main()
