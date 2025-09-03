import os, json, itertools, random, re, argparse
from datetime import datetime
from glob import glob

# Apple Silicon MPS bellek üst sınırı kapatma (OOM riskini azaltır)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
)

# -----------------------
# Reproducibility & device
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# -----------------------
# CLI args (paths & model)
# -----------------------
ap = argparse.ArgumentParser()
ap.add_argument("--train_csv", default="data/train.csv")
ap.add_argument("--val_csv",   default="data/val.csv")
ap.add_argument("--test_csv",  default="data/test.csv")
ap.add_argument("--model_name", default="bert-base-uncased")
# >>> NEW: yalnızca refit + raporlama için
ap.add_argument("--only_refit", action="store_true",
                help="Grid aramasını atla; grid_summary.csv'den en iyiyi al, Train+Val üzerinde refit yap ve Test'i bir kez değerlendir.")
cli = ap.parse_args()

TRAIN_CSV = cli.train_csv
VAL_CSV   = cli.val_csv
TEST_CSV  = cli.test_csv
MODEL_NAME = cli.model_name

# -----------------------
# Paths & ACTIVE_DIR mantığı
# -----------------------
ROOT_DIR = "results/phase2/phase2_3_hparam"
os.makedirs(ROOT_DIR, exist_ok=True)
ACTIVE_FILE = os.path.join(ROOT_DIR, "ACTIVE_DIR.txt")

def _latest_timestamp_dir(root: str):
    cand = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and re.fullmatch(r"\d{8}_\d{6}", d):
            cand.append(d)
    if not cand:
        return None
    cand.sort()
    return os.path.join(root, cand[-1])

if os.path.exists(ACTIVE_FILE):
    with open(ACTIVE_FILE, "r") as f:
        candidate = f.read().strip()
    if candidate and os.path.isdir(candidate):
        BASE_DIR = candidate
    else:
        latest = _latest_timestamp_dir(ROOT_DIR)
        if latest:
            BASE_DIR = latest
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            BASE_DIR = os.path.join(ROOT_DIR, timestamp)
            os.makedirs(BASE_DIR, exist_ok=True)
else:
    latest = _latest_timestamp_dir(ROOT_DIR)
    if latest:
        BASE_DIR = latest
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        BASE_DIR = os.path.join(ROOT_DIR, timestamp)
        os.makedirs(BASE_DIR, exist_ok=True)

with open(ACTIVE_FILE, "w") as f:
    f.write(BASE_DIR)

print(f"[Phase 2.3] Working directory -> {BASE_DIR}")

# -----------------------
# Data loading
# -----------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "text_bert" in df.columns, "CSV must contain 'text_bert'"
    assert "sentiment" in df.columns, "CSV must contain 'sentiment'"
    df = df.dropna(subset=["text_bert", "sentiment"]).copy()
    df["label"] = df["sentiment"].astype(str).str.lower().map(LABEL2ID)
    df = df[df["label"].isin([0,1,2])].reset_index(drop=True)
    return df

train_df = load_df(TRAIN_CSV)
val_df   = load_df(VAL_CSV)
test_df  = load_df(TEST_CSV)

# -----------------------
# Tokenisation helpers
# -----------------------
def build_datasets(model_name: str, max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tok(batch):
        return tokenizer(batch["text_bert"], truncation=True, max_length=max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def to_ds(df: pd.DataFrame):
        ds = Dataset.from_pandas(
            df[["text_bert","label"]].rename(columns={"label":"labels"}),
            preserve_index=False
        )
        ds = ds.map(tok, batched=True, remove_columns=["text_bert"])
        return ds

    return tokenizer, to_ds(train_df), to_ds(val_df), to_ds(test_df), collator

# -----------------------
# Metrics
# -----------------------
def compute_metrics(eval_pred):
    if isinstance(eval_pred, tuple) and len(eval_pred) == 2:
        logits, labels = eval_pred
    else:
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "macro_precision": p_m, "macro_recall": r_m, "macro_f1": f_m, "weighted_f1": f_w}

def save_confusion_matrix_csv(y_true, y_pred, out_csv: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    pd.DataFrame(cm, index=[ID2LABEL[i] for i in [0,1,2]],
                 columns=[ID2LABEL[i] for i in [0,1,2]]).to_csv(out_csv)

def latest_checkpoint(run_dir: str):
    paths = glob(os.path.join(run_dir, "checkpoint-*"))
    if not paths:
        return None
    def _step(p):
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1
    paths.sort(key=_step)
    return paths[-1]

def read_run_eval(run_dir: str):
    p = os.path.join(run_dir, "run_eval.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

# -----------------------
# Search space
# -----------------------
SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size":    [8],
    "max_len":       [128, 256],
    "epochs":        [3, 4],
}
CANDIDATES = list(itertools.product(
    SEARCH_SPACE["learning_rate"],
    SEARCH_SPACE["batch_size"],
    SEARCH_SPACE["max_len"],
    SEARCH_SPACE["epochs"]
))

results_rows = []
best = {"macro_f1": -1.0, "cfg": None, "val_dir": None}

# -----------------------
# Grid search (skip if --only_refit)
# -----------------------
if not cli.only_refit:
    for (lr, bs, max_len, epochs) in CANDIDATES:
        run_tag = f"lr{lr}_bs{bs}_len{max_len}_ep{epochs}"
        run_dir = os.path.join(BASE_DIR, run_tag)
        os.makedirs(run_dir, exist_ok=True)

        done_flag = os.path.join(run_dir, "DONE.flag")
        if os.path.exists(done_flag):
            prev = read_run_eval(run_dir)
            if prev:
                results_rows.append({
                    "lr": lr, "batch_size": bs, "max_len": max_len, "epochs": epochs,
                    "eval_accuracy": float(prev.get("accuracy", 0.0)),
                    "eval_macro_precision": float(prev.get("macro_precision", 0.0)),
                    "eval_macro_recall": float(prev.get("macro_recall", 0.0)),
                    "eval_macro_f1": float(prev.get("macro_f1", 0.0)),
                    "eval_weighted_f1": float(prev.get("weighted_f1", 0.0)),
                    "run_dir": run_dir
                })
                if prev.get("macro_f1", 0.0) > best["macro_f1"]:
                    best = {"macro_f1": prev["macro_f1"], "cfg": (lr, bs, max_len, epochs), "val_dir": run_dir}
            print(f"[SKIP] {run_tag} (DONE.flag)")
            pd.DataFrame(results_rows).sort_values(
                by="eval_macro_f1", ascending=False
            ).to_csv(os.path.join(BASE_DIR, "grid_summary.csv"), index=False)
            continue

        tokenizer, train_ds, val_ds, test_ds, collator = build_datasets(MODEL_NAME, max_len)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        )

        args = TrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            learning_rate=lr,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            dataloader_num_workers=0,
            seed=SEED
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
            optimizers=(AdamW(model.parameters(), lr=lr), None),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        ckpt = latest_checkpoint(run_dir)
        if ckpt:
            print(f"[RESUME] {run_tag} -> {ckpt}")
            trainer.train(resume_from_checkpoint=ckpt)
        else:
            print(f"[RUN ] {run_tag} (fresh)")
            trainer.train()

        eval_out = trainer.evaluate()

        metrics_clean = {
            "loss": float(eval_out.get("eval_loss", 0.0)),
            "accuracy": float(eval_out.get("eval_accuracy", 0.0)),
            "macro_precision": float(eval_out.get("eval_macro_precision", 0.0)),
            "macro_recall": float(eval_out.get("eval_macro_recall", 0.0)),
            "macro_f1": float(eval_out.get("eval_macro_f1", 0.0)),
            "weighted_f1": float(eval_out.get("eval_weighted_f1", 0.0)),
        }
        with open(os.path.join(run_dir, "run_eval.json"), "w") as f:
            json.dump(metrics_clean, f, indent=2)

        results_rows.append({
            "lr": lr, "batch_size": bs, "max_len": max_len, "epochs": epochs,
            "eval_accuracy": metrics_clean["accuracy"],
            "eval_macro_precision": metrics_clean["macro_precision"],
            "eval_macro_recall": metrics_clean["macro_recall"],
            "eval_macro_f1": metrics_clean["macro_f1"],
            "eval_weighted_f1": metrics_clean["weighted_f1"],
            "run_dir": run_dir
        })

        if metrics_clean["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": metrics_clean["macro_f1"], "cfg": (lr, bs, max_len, epochs), "val_dir": run_dir}

        with open(done_flag, "w") as f:
            f.write("ok")

        pd.DataFrame(results_rows).sort_values(
            by="eval_macro_f1", ascending=False
        ).to_csv(os.path.join(BASE_DIR, "grid_summary.csv"), index=False)

    grid_df = pd.DataFrame(results_rows).sort_values(by="eval_macro_f1", ascending=False)
    grid_df.to_csv(os.path.join(BASE_DIR, "grid_summary.csv"), index=False)

# Eğer sadece refit isteniyorsa, grid_summary'den en iyiyi oku
if cli.only_refit:
    grid_path = os.path.join(BASE_DIR, "grid_summary.csv")
    if not os.path.exists(grid_path):
        # fallback: run klasörlerindeki run_eval.json'ları topla
        rows = []
        for d in sorted(os.listdir(BASE_DIR)):
            rd = os.path.join(BASE_DIR, d)
            if os.path.isdir(rd) and d.startswith("lr"):
                ev = read_run_eval(rd)
                if ev:
                    # lr, bs, len, ep'i tag'den parse et
                    m = re.match(r"lr(?P<lr>[\de\.-]+)_bs(?P<bs>\d+)_len(?P<len>\d+)_ep(?P<ep>\d+)", d)
                    if m:
                        rows.append({
                            "lr": float(m.group("lr")), "batch_size": int(m.group("bs")),
                            "max_len": int(m.group("len")), "epochs": int(m.group("ep")),
                            "eval_macro_f1": float(ev.get("macro_f1", 0.0)), "run_dir": rd
                        })
        if not rows:
            raise RuntimeError("En iyi konfigürasyonu bulamadım (grid_summary yok).")
        grid_df = pd.DataFrame(rows).sort_values(by="eval_macro_f1", ascending=False)
        grid_df.to_csv(grid_path, index=False)
    else:
        grid_df = pd.read_csv(grid_path)

    top = grid_df.sort_values(by="eval_macro_f1", ascending=False).iloc[0]
    best = {"macro_f1": float(top["eval_macro_f1"]),
            "cfg": (float(top["lr"]), int(top["batch_size"]), int(top["max_len"]), int(top["epochs"])),
            "val_dir": top["run_dir"]}

# -----------------------
# Refit best on Train+Val, evaluate on Test (single-shot)
# -----------------------
if best["cfg"] is None:
    raise RuntimeError("No completed runs; cannot refit.")

best_lr, best_bs, best_len, best_epochs = best["cfg"]
refit_tag = f"BEST_refit_lr{best_lr}_bs{best_bs}_len{best_len}_ep{best_epochs}"
refit_dir = os.path.join(BASE_DIR, refit_tag)
os.makedirs(refit_dir, exist_ok=True)

# Merge Train + Val
trainval_df = pd.concat([train_df, val_df], ignore_index=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tok_tv(batch):
    return tokenizer(batch["text_bert"], truncation=True, max_length=best_len)

def to_ds(df: pd.DataFrame):
    ds = Dataset.from_pandas(df[["text_bert","label"]].rename(columns={"label":"labels"}), preserve_index=False)
    return ds.map(tok_tv, batched=True, remove_columns=["text_bert"])

trainval_ds = to_ds(trainval_df)
test_ds     = to_ds(test_df)
collator    = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)

# >>> IMPORTANT: Tez disiplinine uygun — refit sırasında test'e bakma
args = TrainingArguments(
    output_dir=refit_dir,
    per_device_train_batch_size=best_bs,
    per_device_eval_batch_size=best_bs,
    learning_rate=best_lr,
    num_train_epochs=best_epochs,
    weight_decay=0.01,

    # Eğitim sırasında evaluation KAPALI (testi asla eval için kullanmıyoruz)
    eval_strategy="no",          # senin sürümde bu isim geçerli
    save_strategy="epoch",       # ister "epoch" bırak (checkpoint alır), ister "no" yap
    save_total_limit=2,

    load_best_model_at_end=False,  # eval yokken "best" kavramı yok

    logging_steps=100,
    report_to="none",
    dataloader_num_workers=0,
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainval_ds,
    # eval_dataset=None  # <-- eğitim sırasında eval yok
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,   # sadece evaluate(test_ds) çağrısında kullanılır
    optimizers=(AdamW(model.parameters(), lr=best_lr), None)
    # callbacks=[]  # EarlyStopping kaldırıldı
)

refit_ckpt = latest_checkpoint(refit_dir)
trainer.train(resume_from_checkpoint=refit_ckpt if refit_ckpt else None)

# Tek sefer evaluation on Test
test_out = trainer.evaluate(test_ds)

# Predictions & reports
pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

rep = classification_report(
    y_true, y_pred,
    target_names=[ID2LABEL[i] for i in [0,1,2]],
    output_dict=True, digits=4
)
pd.DataFrame(rep).transpose().to_csv(os.path.join(refit_dir, "bert_test_report.csv"))

acc = accuracy_score(y_true, y_pred)
p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

overview = {
    "accuracy": acc,
    "macro_precision": p_m, "macro_recall": r_m, "macro_f1": f_m,
    "weighted_f1": f_w,
    "best_lr": best_lr, "best_batch_size": best_bs,
    "best_max_len": best_len, "best_epochs": best_epochs
}
pd.DataFrame([overview]).to_csv(os.path.join(refit_dir, "bert_overview_metrics_test.csv"), index=False)

save_confusion_matrix_csv(y_true, y_pred, os.path.join(refit_dir, "bert_confusion_matrix_test.csv"))

pd.DataFrame({
    "text_bert": test_df["text_bert"],
    "sentiment": test_df["sentiment"],
    "pred_label_id": y_pred,
    "pred_label": [ID2LABEL[i] for i in y_pred]
}).to_csv(os.path.join(refit_dir, "bert_test_predictions.csv"), index=False)

with open(os.path.join(BASE_DIR, "best_config.json"), "w") as f:
    json.dump({
        "best_validation_macro_f1": best["macro_f1"],
        "best_config": {
            "learning_rate": best_lr, "batch_size": best_bs, "max_len": best_len, "epochs": best_epochs
        },
        "best_val_run_dir": best["val_dir"],
        "final_test_dir": refit_dir
    }, f, indent=2)

# -----------------------
# Thesis tables (CSV) — grid summary + best test
# -----------------------
if os.path.exists(os.path.join(BASE_DIR, "grid_summary.csv")):
    table1 = pd.read_csv(os.path.join(BASE_DIR, "grid_summary.csv"))[[
        "lr","batch_size","max_len","epochs",
        "eval_accuracy","eval_macro_precision","eval_macro_recall","eval_macro_f1","eval_weighted_f1","run_dir"
    ]].rename(columns={
        "lr":"Learning Rate", "batch_size":"Batch Size", "max_len":"Max Seq. Len", "epochs":"Epochs",
        "eval_accuracy":"Accuracy", "eval_macro_precision":"Macro Precision",
        "eval_macro_recall":"Macro Recall", "eval_macro_f1":"Macro F1", "eval_weighted_f1":"Weighted F1",
        "run_dir":"Run Dir"
    }).sort_values(by="Macro F1", ascending=False)
    table1_path = os.path.join(BASE_DIR, "table_hparam_search_summary.csv")
    table1.to_csv(table1_path, index=False)

table2 = pd.DataFrame([{
    "Learning Rate": best_lr,
    "Batch Size": best_bs,
    "Max Seq. Len": best_len,
    "Epochs": best_epochs,
    "Accuracy": overview["accuracy"],
    "Macro Precision": overview["macro_precision"],
    "Macro Recall": overview["macro_recall"],
    "Macro F1": overview["macro_f1"],
    "Weighted F1": overview["weighted_f1"]
}])
table2_path = os.path.join(BASE_DIR, "table_best_config_test_performance.csv")
table2.to_csv(table2_path, index=False)

# -----------------------
# EXTRA (tez çıktıları): Class-wise table + CM PNG + LaTeX/Markdown
# -----------------------
rep_df = pd.read_csv(os.path.join(refit_dir, "bert_test_report.csv"))
want = rep_df.loc[rep_df["Unnamed: 0"].isin(["negative","neutral","positive"]),
                  ["Unnamed: 0","precision","recall","f1-score","support"]].copy()
want.columns = ["Class","Precision","Recall","F1","Support"]
want.to_csv(os.path.join(refit_dir, "TABLE_B_classwise_metrics.csv"), index=False)

# LaTeX (Table B)
rows = "\n".join([f"{r.Class} & {r.Precision:.4f} & {r.Recall:.4f} & {r.F1:.4f} & {int(r.Support)} \\\\"
                   for r in want.itertuples()])
tableB_tex = (
    "\\begin{table}[h]\\centering\\caption{Class-wise metrics on the test set.}\n"
    "\\begin{tabular}{lcccc}\\hline\n"
    "Class & Precision & Recall & F1 & Support \\\\ \\hline\n"
    f"{rows}\n\\hline\n\\end{tabular}\\label{{tab:bert_classwise}}\\end{table}\n"
)
with open(os.path.join(refit_dir, "TABLE_B_classwise_metrics.tex"), "w") as f:
    f.write(tableB_tex)

# Table A (overall) — LaTeX + Markdown
acc  = float(overview["accuracy"]); mp = float(overview["macro_precision"])
mr   = float(overview["macro_recall"]); mf1 = float(overview["macro_f1"]); wf1 = float(overview["weighted_f1"])
tableA_tex = (
    "\\begin{table}[h]\\centering\\caption{Test performance of BERT (best config).}\n"
    "\\begin{tabular}{ccccc}\\hline\n"
    "Accuracy & Macro Prec. & Macro Rec. & Macro F1 & Weighted F1 \\\\ \\hline\n"
    f"{acc:.4f} & {mp:.4f} & {mr:.4f} & {mf1:.4f} & {wf1:.4f} \\\\ \\hline\n"
    "\\end{tabular}\\label{tab:bert_test}\\end{table}\n"
)
with open(os.path.join(refit_dir, "TABLE_A_test_overview.tex"), "w") as f:
    f.write(tableA_tex)
with open(os.path.join(refit_dir, "TABLE_A_test_overview.md"), "w") as f:
    f.write(
        f"**Table A. Test performance of the best BERT configuration**  \n"
        f"Accuracy: **{acc:.4f}**, Macro-Precision: **{mp:.4f}**, Macro-Recall: **{mr:.4f}**, "
        f"Macro-F1: **{mf1:.4f}**, Weighted-F1: **{wf1:.4f}**\n"
    )

# Confusion Matrix PNG
try:
    import matplotlib.pyplot as plt
    cm_df = pd.read_csv(os.path.join(refit_dir, "bert_confusion_matrix_test.csv"), index_col=0)
    fig = plt.figure()
    plt.imshow(cm_df.values, interpolation="nearest")
    plt.title("Confusion Matrix (BERT — Test)")
    plt.xticks(ticks=range(3), labels=list(cm_df.columns))
    plt.yticks(ticks=range(3), labels=list(cm_df.index))
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(cm_df.values[i, j])), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    out_png = os.path.join(refit_dir, "FIG_confusion_matrix_test.png")
    plt.savefig(out_png, dpi=300)
    print(f"- Confusion matrix figure: {out_png}")
except Exception as e:
    print(f"[INFO] Confusion matrix PNG skipped: {e}")

print("\n=== Phase 2.3 complete ===")
print(f"Best (val Macro-F1): {best['macro_f1']:.4f} with "
      f"lr={best_lr}, bs={best_bs}, len={best_len}, epochs={best_epochs}")
print(f"- Grid summary: {os.path.join(BASE_DIR, 'grid_summary.csv')}")
print(f"- Final test dir: {refit_dir}")
print(f"- Tables: {os.path.join(refit_dir, 'TABLE_A_test_overview.tex')}, "
      f"{os.path.join(refit_dir, 'TABLE_B_classwise_metrics.tex')}")
