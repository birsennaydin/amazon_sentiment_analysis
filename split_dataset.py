import pandas as pd
from sklearn.model_selection import train_test_split

# 1. CSV'yi oku
df = pd.read_csv("data/amazon_electronics_prepared.csv")

# 2. Stratified Split (train + temp)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,  # %70 train
    stratify=df['sentiment'],
    random_state=42
)

# 3. Stratified Split (validation + test)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,  # %15 val, %15 test
    stratify=temp_df['sentiment'],
    random_state=42
)

# 4. Dosyalara kaydet
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
print(train_df['sentiment'].value_counts())
print(val_df['sentiment'].value_counts())
print(test_df['sentiment'].value_counts())
