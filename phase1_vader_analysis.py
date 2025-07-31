import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the prepared data for Phase 1
df = pd.read_csv("amazon_phase1_ready.csv")
print(f"Data loaded! Number of rows: {len(df)}")
print(df.head())

# 2. Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# 3. Calculate VADER sentiment scores
df["vader_score"] = df["text"].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# 4. Label the scores as positive/negative/neutral
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

# 6. Plot the confusion matrix
labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("VADER Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("phase1_vader_confusion_matrix.png")
plt.show()

# 7. Save the results
output_file = "phase1_vader_results.csv"
df.to_csv(output_file, index=False)
print(f"\nPhase 1 completed! Results have been saved to '{output_file}'.")
