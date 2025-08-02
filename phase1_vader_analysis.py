import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 1. Load data
df = pd.read_csv("amazon_phase1_ready.csv")
print(f"Data loaded! Number of rows: {len(df)}")

# 2. Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# 3. Calculate VADER sentiment scores
df["vader_score"] = df["text"].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# 4. Convert score to sentiment label
def vader_to_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_sentiment"] = df["vader_score"].apply(vader_to_label)

# 5. Performance evaluation
y_true = df["true_sentiment"]
y_pred = df["vader_sentiment"]

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print("\nVADER Performance Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 6. Confusion matrix
labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Normalize for percentage heatmap
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 7. Create timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 8. Plot confusion matrix (count)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("VADER Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"vader_confusion_matrix_{timestamp}.png", dpi=300)
plt.close()

# 9. Plot confusion matrix (percentage)
plt.figure(figsize=(6,4))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("VADER Confusion Matrix (%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"vader_confusion_matrix_percent_{timestamp}.png", dpi=300)
plt.close()

# 10. Class summary (Toplam, Doğru, Accuracy per class)
summary = []
for i, label in enumerate(labels):
    total = cm[i].sum()
    correct = cm[i,i]
    acc_class = correct / total if total > 0 else 0
    summary.append([label, total, correct, acc_class])

class_summary_df = pd.DataFrame(summary, columns=["class", "total", "correct", "accuracy_per_class"])
class_summary_df.loc[len(class_summary_df)] = ["TOTAL", cm.sum(), np.trace(cm), accuracy]
class_summary_csv = f"vader_class_summary_{timestamp}.csv"
class_summary_df.to_csv(class_summary_csv, index=False)

# 11. Plot correct/incorrect bar chart
correct_counts = np.diag(cm)
incorrect_counts = cm.sum(axis=1) - correct_counts

plt.figure(figsize=(7,5))
x = np.arange(len(labels))
bar_width = 0.35

plt.bar(x - bar_width/2, correct_counts, width=bar_width, label="Correct")
plt.bar(x + bar_width/2, incorrect_counts, width=bar_width, label="Incorrect")

plt.xticks(x, labels)
plt.ylabel("Count")
plt.title("Correct vs Incorrect Predictions per Class")
plt.legend()
plt.savefig(f"vader_correct_incorrect_bar_{timestamp}.png", dpi=300)
plt.close()

# 12. Save first CSV with summarized text
df_summary = df.copy()
df_summary["text_summary"] = df_summary["text"].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
df_summary_export = df_summary[["text_summary", "true_sentiment", "vader_sentiment"]]
summary_csv = f"vader_results_summary_{timestamp}.csv"
df_summary_export.to_csv(summary_csv, index=False)

print(f"\nAll files saved with timestamp {timestamp}")
print(f"1) {summary_csv}")
print(f"2) {class_summary_csv}")
print(f"3) Confusion Matrices and Bar Graphs saved as PNG")
