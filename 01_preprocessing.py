from datasets import load_dataset
import pandas as pd
import re

# 1. Veri İndirme
print("Downloading Amazon Electronics data...")
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    streaming=True,
    trust_remote_code=True
)
stream_iter = iter(dataset['full'])

print("Fetching the first 1,000,000 records...")
raw_data = [next(stream_iter) for _ in range(1_000_000)]
df = pd.DataFrame(raw_data)

# 2. Temizlik
df = df[['rating', 'text']].rename(columns={'text': 'text_raw'})
df.dropna(subset=['text_raw'], inplace=True)
df['text_raw'] = df['text_raw'].str.strip()
df = df[df['text_raw'] != ""]
df = df[df['text_raw'].str.len() > 2]
df.drop_duplicates(subset=['text_raw'], inplace=True)

print(f"After cleaning: {len(df)} records remain.")

# 3. Sentiment Label
def label_sentiment(rating):
    if rating <= 2: return "negative"
    elif rating == 3: return "neutral"
    else: return "positive"

df['sentiment'] = df['rating'].apply(label_sentiment)

# 4. Her sınıftan 17K (3*17K ~ 51K)
print("Balancing dataset to 17k per class...")
final_df = (
    df.groupby('sentiment', group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 17000), random_state=42))
)

# 5. Preprocessing
def preprocess_vader(text):
    if pd.isna(text): return ""
    return str(text).lower().strip()

def preprocess_bert(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

final_df['text_vader'] = final_df['text_raw'].apply(preprocess_vader)
final_df['text_bert'] = final_df['text_raw'].apply(preprocess_bert)

# 6. Son Temizlik
final_df.dropna(subset=['text_raw', 'text_vader', 'text_bert'], inplace=True)
final_df = final_df[
    (final_df['text_raw'].str.strip() != "") &
    (final_df['text_vader'].str.strip() != "") &
    (final_df['text_bert'].str.strip() != "")
]

# 7. Kaydet
output_file = "data/amazon_electronics_50k.csv"
final_df.to_csv(output_file, index=False)

print(final_df['sentiment'].value_counts())
print(final_df.isnull().sum())
print(f"Dataset is ready and saved to '{output_file}'")
