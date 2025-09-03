import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# -----------------------------
# Global label config
# -----------------------------
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def datetime_now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def latest_checkpoint(dir_path: str):
    """Return latest checkpoint path inside dir_path or None."""
    if not os.path.isdir(dir_path):
        return None
    names = [n for n in os.listdir(dir_path) if n.startswith("checkpoint-")]
    if not names:
        return None
    def step(n):
        m = re.search(r"checkpoint-(\d+)", n)
        return int(m.group(1)) if m else -1
    best = sorted(names, key=step)[-1]
    return os.path.join(dir_path, best)

# -----------------------------
# Dataset
# -----------------------------
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels  # np.array of ints (0,1,2)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

# -----------------------------
# Metrics & Plots
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}

def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    plt.figure(figsize=(6,5))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

def save_fold_outputs(root_dir, fold_idx, y_true, y_pred):
    # overview metrics
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    pd.DataFrame([[f"Fold-{fold_idx}", acc, p, r, f1, w_f1]],
                 columns=["Dataset","Accuracy","Macro_Precision","Macro_Recall","Macro_F1","Weighted_F1"]
                 ).to_csv(os.path.join(root_dir, f"bert_overview_metrics_fold{fold_idx}.csv"), index=False)

    # classification report
    cr = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, digits=4)
    pd.DataFrame(cr).transpose().to_csv(os.path.join(root_dir, f"bert_test_report_fold{fold_idx}.csv"))

    # confusion matrix csv
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        os.path.join(root_dir, f"bert_confusion_matrix_fold{fold_idx}.csv")
    )

    # class summary
    rows=[]
    for i, lab in enumerate(LABELS):
        total = cm[i].sum(); correct = cm[i, i]
        rows.append([lab, int(total), int(correct), (correct/total if total>0 else 0.0)])
    rows.append(["TOTAL", int(cm.sum()), int((np.array(y_true)==np.array(y_pred)).sum()), acc])
    pd.DataFrame(rows, columns=["Class","Total","Correct","Accuracy per Class"]
                 ).to_csv(os.path.join(root_dir, f"bert_class_summary_fold{fold_idx}.csv"), index=False)

    # confusion plot
    plot_confusion_matrix(y_true, y_pred,
                          f"Confusion Matrix – BERT (Fold {fold_idx})",
                          os.path.join(root_dir, f"bert_confusion_matrix_fold{fold_idx}.png"))

def aggregate_overview(root_dir, k):
    dfs=[]
    for i in range(1, k+1):
        p = os.path.join(root_dir, f"bert_overview_metrics_fold{i}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return None, None
    big = pd.concat(dfs, ignore_index=True)
    metrics = ["Accuracy","Macro_Precision","Macro_Recall","Macro_F1","Weighted_F1"]
    mean_vals = big[metrics].mean(); std_vals = big[metrics].std()
    agg = pd.DataFrame([["K-Fold Mean", *mean_vals.values], ["K-Fold Std", *std_vals.values]],
                       columns=["Dataset", *metrics])
    agg.to_csv(os.path.join(root_dir, "bert_overview_metrics_kfold_summary.csv"), index=False)
    return big, agg

# -----------------------------
# Data loading
# -----------------------------
def load_dataset(full_csv):
    df = pd.read_csv(full_csv)
    if "text_bert" not in df.columns:
        if "text_raw" in df.columns:
            df["text_bert"] = df["text_raw"].astype(str)
        else:
            raise KeyError("Dataset must include 'text_bert' (or at least 'text_raw').")
    if "sentiment" not in df.columns:
        raise KeyError("Dataset must include 'sentiment' column.")

    df = df[["text_bert","sentiment"]].dropna().reset_index(drop=True)
    # safe map (strings -> ids)
    df["sentiment"] = df["sentiment"].astype(str).str.lower().map(LABEL2ID)
    df = df[df["sentiment"].isin([0,1,2])].reset_index(drop=True)

    texts = df["text_bert"].astype(str).tolist()
    labels = df["sentiment"].astype(int).to_numpy()
    return texts, labels

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--run_dir", type=str, default="", help="Sabit çıktı klasörü; verilirse timestamp kullanılmaz.")
    parser.add_argument("--limit_rows", type=int, default=0, help="Hızlı test için veri kırpma (0=kapalı).")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(42)

    # output root
    if args.run_dir:
        out_root = args.run_dir
    else:
        out_root = os.path.join("results","phase2","phase2_2_kfold", datetime_now())
    ensure_dir(out_root)
    print(f"[Phase 2.2] Outputs -> {out_root}")

    # load data
    texts, labels = load_dataset(args.full_csv)
    if args.limit_rows and args.limit_rows > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=min(args.limit_rows, len(texts)), replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]
        print(f"[DEBUG] Using subset of rows: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    all_true, all_pred = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(texts, labels), start=1):
        # fold paths & flags
        fold_dir = os.path.join(out_root, f"fold_{fold_idx}")
        ensure_dir(fold_dir)
        fold_done_flag = os.path.join(out_root, f"fold_{fold_idx}_done.txt")

        # skip finished folds
        if os.path.exists(fold_done_flag):
            print(f"[INFO] Fold {fold_idx} already completed. Skipping.")
            continue

        print(f"\n=== Fold {fold_idx}/{args.k} ===")
        X_train = [texts[i] for i in tr_idx]; y_train = labels[tr_idx]
        X_test  = [texts[i] for i in te_idx]; y_test  = labels[te_idx]

        train_ds = ReviewDataset(X_train, y_train, tokenizer, args.max_len)
        test_ds  = ReviewDataset(X_test,  y_test,  tokenizer, args.max_len)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        )

        training_args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",           # checkpoint yaz
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            logging_steps=50,
            report_to="none",
            dataloader_num_workers=0,        # M1/macOS için daha stabil
            gradient_accumulation_steps=2    # bs=8 ile efektif 16; bs=16 ise efektif 32 olur (gerekirse 1 yap)
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,            # K-Fold: test fold üzerinde rapor
            compute_metrics=compute_metrics
        )

        # resume inside fold if checkpoint exists
        ckpt = latest_checkpoint(fold_dir)
        trainer.train(resume_from_checkpoint=ckpt if ckpt else None)

        preds = trainer.predict(test_ds)
        y_pred = np.argmax(preds.predictions, axis=-1)

        save_fold_outputs(out_root, fold_idx, y_test, y_pred)
        # mark fold as done
        with open(fold_done_flag, "w") as f:
            f.write("done")

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    # aggregate overview
    _, agg = aggregate_overview(out_root, args.k)
    if agg is not None:
        print("\n[K-Fold Summary]")
        print(agg.to_string(index=False))

    # aggregated confusion
    if all_true and all_pred:
        plot_confusion_matrix(np.array(all_true), np.array(all_pred),
                              "Aggregated Confusion Matrix",
                              os.path.join(out_root, "bert_confusion_matrix_aggregated.png"))

    print(f"\nAll results saved under: {out_root}")

if __name__ == "__main__":
    main()
