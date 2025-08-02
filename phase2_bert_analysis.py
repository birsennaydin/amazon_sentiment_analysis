import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. CSV'yi oku ve hazırla
# =========================
df = pd.read_csv("amazon_phase1_ready.csv")

# BERT için gerekli kolonlar
df = df.rename(columns={"text": "review_text", "true_sentiment": "sentiment"})
df = df.dropna(subset=["review_text", "sentiment"])
print("Dataset shape:", df.shape)

# Label encode sentiments (negative=0, neutral=1, positive=2)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["sentiment"])

# Train / Validation / Test böl
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

print(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

# =========================
# 2. HuggingFace Dataset formatına çevir
# =========================
train_dataset = Dataset.from_pandas(train_df[["review_text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["review_text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["review_text", "label"]])

# =========================
# 3. Tokenizer yükle ve tokenize et
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["review_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Trainer'ın ihtiyacı olan format
train_dataset = train_dataset.remove_columns(["review_text"])
val_dataset = val_dataset.remove_columns(["review_text"])
test_dataset = test_dataset.remove_columns(["review_text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

# =========================
# 4. Model ve Trainer ayarları
# =========================
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Evaluation metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./bert_sentiment_results",
    eval_strategy="epoch",          # ✅ doğru parametre
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=50
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# =========================
# 5. Modeli eğit
# =========================
trainer.train()

# =========================
# 6. Test setinde değerlendirme
# =========================
results = trainer.evaluate(test_dataset)
print("Test Results:", results)
