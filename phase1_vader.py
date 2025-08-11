# ===============================
# Phase 1.4 – Threshold Optimization (Grid Search)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ===============================
# 1. VADER Setup with Custom Lexicon
# ===============================
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

custom_lexicon = {
    "excellent": 3.0, "awesome": 2.5, "legendary": 2.0, "superb": 2.0,
    "verygood": 2.5, "lovedit": 2.5, "fantastic": 2.5,
    "terrible": -3.0, "disgusting": -3.0, "awful": -2.5, "horrible": -2.5,
    "catastrophic": -2.5, "hate": -2.5, "bad": -2.0
}
sia.lexicon.update(custom_lexicon)

# ===============================
# 2. Load Train and Test CSV
# ===============================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print("Train:", train_df.shape, "Test:", test_df.shape)

# ===============================
# 2.1 Preprocessing
# ===============================
slang_dict = {
    "u": "you", "ur": "your", "idk": "i don't know", "omg": "oh my god",
    "lol": "laughing out loud", "lmao": "laughing my ass off",
    "wtf": "what the fuck", "btw": "by the way", "imo": "in my opinion",
    "imho": "in my humble opinion"
}

def custom_preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized_words)

for df in [train_df, test_df]:
    if 'text_vader' in df.columns:
        df['text_vader'] = df['text_vader'].apply(custom_preprocess)
    elif 'text' in df.columns:
        df['text_vader'] = df['text'].apply(custom_preprocess)
    elif 'text_raw' in df.columns:
        df['text_vader'] = df['text_raw'].apply(custom_preprocess)
    else:
        raise KeyError("CSV must have 'text_vader', 'text', or 'text_raw' column!")

    if 'text_raw' not in df.columns:
        df['text_raw'] = df['text_vader']

# ===============================
# 3. VADER Prediction Function
# ===============================
def vader_predict(text, pos_th, neg_th):
    if pd.isna(text) or not str(text).strip():
        return "neutral"
    score = sia.polarity_scores(str(text))
    compound = score['compound']
    if compound >= pos_th:
        return "positive"
    elif compound <= neg_th:
        return "negative"
    else:
        return "neutral"

# ===============================
# 4. Grid Search for Best Thresholds
# ===============================
def grid_search_best_threshold(df, true_col, text_col):
    best_f1 = -1
    best_pos, best_neg = 0, 0

    thresholds = np.arange(0.05, 0.55, 0.05)  # Search range
    for pos_th in thresholds:
        for neg_th in -thresholds[::-1]:  # e.g., -0.05 to -0.5
            preds = df[text_col].apply(lambda x: vader_predict(x, pos_th, neg_th))
            macro_f1 = f1_score(df[true_col], preds, average='macro')
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_pos, best_neg = pos_th, neg_th

    return best_pos, best_neg, best_f1

best_pos, best_neg, best_f1 = grid_search_best_threshold(train_df, 'sentiment', 'text_vader')
print(f"\nBest Thresholds: positive >= {best_pos:.2f}, negative <= {best_neg:.2f} (Train Macro F1: {best_f1:.4f})")

# ===============================
# 5. Final Predictions with Best Threshold
# ===============================
train_df['vader_pred'] = train_df['text_vader'].apply(lambda x: vader_predict(x, best_pos, best_neg))
test_df['vader_pred'] = test_df['text_vader'].apply(lambda x: vader_predict(x, best_pos, best_neg))

# ===============================
# 6. Compute Performance
# ===============================
train_acc = accuracy_score(train_df['sentiment'], train_df['vader_pred'])
test_acc = accuracy_score(test_df['sentiment'], test_df['vader_pred'])

train_macro_precision = precision_score(train_df['sentiment'], train_df['vader_pred'], average='macro', zero_division=0)
train_macro_recall = recall_score(train_df['sentiment'], train_df['vader_pred'], average='macro', zero_division=0)
train_macro_f1 = f1_score(train_df['sentiment'], train_df['vader_pred'], average='macro')
train_weighted_f1 = f1_score(train_df['sentiment'], train_df['vader_pred'], average='weighted')

test_macro_precision = precision_score(test_df['sentiment'], test_df['vader_pred'], average='macro', zero_division=0)
test_macro_recall = recall_score(test_df['sentiment'], test_df['vader_pred'], average='macro', zero_division=0)
test_macro_f1 = f1_score(test_df['sentiment'], test_df['vader_pred'], average='macro')
test_weighted_f1 = f1_score(test_df['sentiment'], test_df['vader_pred'], average='weighted')

train_report = classification_report(train_df['sentiment'], train_df['vader_pred'], output_dict=True, digits=4)
test_report = classification_report(test_df['sentiment'], test_df['vader_pred'], output_dict=True, digits=4)

train_report_df = pd.DataFrame(train_report).transpose()
test_report_df = pd.DataFrame(test_report).transpose()

print("\n=== TRAIN PERFORMANCE ===")
print(f"Accuracy: {train_acc:.4f}")
print(train_report_df)

print("\n=== TEST PERFORMANCE ===")
print(f"Accuracy: {test_acc:.4f}")
print(test_report_df)

# ===============================
# 7. Class-level Summary
# ===============================
summary_list = []
for label in ["negative", "neutral", "positive"]:
    total = len(test_df[test_df['sentiment'] == label])
    correct = np.sum((test_df['sentiment'] == label) & (test_df['vader_pred'] == label))
    summary_list.append([label, total, correct, correct / total if total > 0 else 0])

summary_df = pd.DataFrame(summary_list, columns=['class', 'total', 'correct', 'accuracy_per_class'])
summary_df.loc[len(summary_df)] = [
    "TOTAL",
    len(test_df),
    np.sum(test_df['sentiment'] == test_df['vader_pred']),
    accuracy_score(test_df['sentiment'], test_df['vader_pred'])
]

print("\n=== SUMMARY TABLE ===")
print(summary_df)

# ===============================
# 8. Confusion Matrix
# ===============================
labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(test_df['sentiment'], test_df['vader_pred'], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\n=== CONFUSION MATRIX ===")
print(cm_df)

# ===============================
# 9. Overview Metrics
# ===============================
overview_df = pd.DataFrame([
    ["Train", train_acc, train_macro_precision, train_macro_recall, train_macro_f1, train_weighted_f1],
    ["Test", test_acc, test_macro_precision, test_macro_recall, test_macro_f1, test_weighted_f1]
], columns=["Dataset", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"])

print("\n=== OVERVIEW METRICS ===")
print(overview_df)

# ===============================
# 10. Save Results
# ===============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "results/phase1/test5"  # <- Phase 1.4 output folder
os.makedirs(output_dir, exist_ok=True)

train_report_df.to_csv(f"{output_dir}/vader_train_report_{timestamp}.csv")
test_report_df.to_csv(f"{output_dir}/vader_test_report_{timestamp}.csv")
summary_df.to_csv(f"{output_dir}/vader_class_summary_{timestamp}.csv", index=False)
cm_df.to_csv(f"{output_dir}/vader_confusion_matrix_{timestamp}.csv")
overview_df.to_csv(f"{output_dir}/vader_overview_metrics_{timestamp}.csv", index=False)

output_csv = f"{output_dir}/vader_predictions_{timestamp}.csv"
test_df[['text_raw', 'sentiment', 'vader_pred']].to_csv(output_csv, index=False)

print(f"\nAll CSVs saved to '{output_dir}' folder with timestamp {timestamp}")

# ===============================
# 11. Graphs
# ===============================
correct_counts = np.diag(cm)
incorrect_counts = cm.sum(axis=1) - correct_counts
x = np.arange(len(labels))

plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, correct_counts, width=0.4, label="Correct", color="#4CAF50")
plt.bar(x + 0.2, incorrect_counts, width=0.4, label="Incorrect", color="#F44336")
plt.xticks(x, labels)
plt.title("Correct vs Incorrect Predictions per Class")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/vader_correct_incorrect_{timestamp}.png")
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{output_dir}/vader_confusion_matrix_{timestamp}.png")
plt.close()

print(f"Graphs saved to '{output_dir}' folder with timestamp {timestamp}")

# ===============================
# 12. Phase Summary
# ===============================
phase_name = "Phase_1_4_Threshold_Optimization"
phase_summary_file = "results/phase_summary.csv"

phase_overview_df = pd.DataFrame([
    [phase_name, "Train", train_acc, train_macro_precision, train_macro_recall, train_macro_f1, train_weighted_f1],
    [phase_name, "Test", test_acc, test_macro_precision, test_macro_recall, test_macro_f1, test_weighted_f1]
], columns=["Phase", "Dataset", "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1"])

if os.path.exists(phase_summary_file):
    old_df = pd.read_csv(phase_summary_file)
    combined_df = pd.concat([old_df, phase_overview_df], ignore_index=True)
    combined_df.to_csv(phase_summary_file, index=False)
else:
    phase_overview_df.to_csv(phase_summary_file, index=False)

print(f"\nPhase summary updated: {phase_summary_file}")
