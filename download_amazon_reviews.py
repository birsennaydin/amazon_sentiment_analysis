from datasets import load_dataset
import pandas as pd

print("Downloading Amazon Electronics data...")

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Electronics",
    streaming=True,
    trust_remote_code=True
)

stream_iter = iter(dataset['full'])

print("Fetching the first 15,000 records...")
first_15000 = [next(stream_iter) for _ in range(15000)]

df = pd.DataFrame(first_15000)

# Keep only the columns we need
columns_to_keep = ['rating', 'title', 'text']
df = df[columns_to_keep]

# Save as CSV
output_file = "amazon_electronics_15000.csv"
df.to_csv(output_file, index=False)
print(f"The first 15,000 records have been saved to '{output_file}'.")
