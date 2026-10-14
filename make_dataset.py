from datasets import load_dataset

ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
df = ds.to_pandas()[["sentence", "label"]]
df = df.rename(columns={"sentence": "text"})
df = df.sample(200, random_state=42)
df.to_csv("data/sample_reviews.csv", index=False)
print(df.head())
print("Saved", len(df), "rows")