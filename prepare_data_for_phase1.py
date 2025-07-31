import pandas as pd

# 1. Load the data
df = pd.read_csv("amazon_electronics_15000.csv")

# 2. Clean missing data
df = df.dropna(subset=["text"])  # text cannot be empty
df = df.drop_duplicates(subset=["text"])  # remove duplicate reviews

# 3. Create Rating → Sentiment label
def map_rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["true_sentiment"] = df["rating"].apply(map_rating_to_sentiment)

# 4. Remove unnecessary columns and reorder
df = df[["rating", "true_sentiment", "title", "text"]]

# 5. Save the prepared data
output_file = "amazon_phase1_ready.csv"
df.to_csv(output_file, index=False)

print(f"✅ Data is ready! '{output_file}' file created. Number of rows: {len(df)}")
print(df.head())
