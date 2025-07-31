# phase1_vader_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. VADER metrics table
metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [0.7916, 0.7930, 0.7916, 0.7923]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("vader_metrics.csv", index=False)
print("VADER Performance Metrics:")
print(metrics_df)
print()

# 2. Confusion Matrix
conf_matrix = np.array([
    [865, 285, 581],
    [258, 94, 642],
    [571, 658, 10416]
])

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title("VADER Confusion Matrix", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.tight_layout()
plt.savefig("vader_confusion_matrix.png", dpi=300)
plt.show()

# 3. English summary for thesis
summary_text = """
Phase 1 – VADER Results:
The VADER analysis conducted on the Amazon Electronics dataset achieved an accuracy of 79.16%.
Precision, Recall, and F1-score values are also around 79%.
Examination of the confusion matrix shows that the model successfully classifies positive reviews but struggles to distinguish neutral reviews.
This limitation arises from the lexicon-based nature of VADER, which has difficulty capturing contextual meaning.
It is expected that the BERT-based model applied in Phase 2 will improve performance, particularly for neutral reviews.
"""

print(summary_text)
print("Metrics CSV and Confusion Matrix PNG saved!")
