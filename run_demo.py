import pandas as pd
from mtester.models import SentimentModel
from mtester.perturbations import Typo, Uppercase, Negation
from mtester.runner import run_tests

data = pd.read_csv("data/sample_reviews.csv")
sentences = data["text"].str.strip().tolist()

model = SentimentModel()
df = run_tests(model, sentences, [Typo(), Uppercase(), Negation()])

print(df.groupby("test")["passed"].agg(["mean", "count"]))
print()
print("Failures:")
print(df[~df["passed"]][["original", "perturbed", "test", "pred_before", "pred_after"]].head(10).to_string())

print()
print("Negation failures:")
neg = df[(df["test"] == "negation") & (~df["passed"])]
print(neg[["original", "perturbed", "pred_before", "pred_after"]].head(10).to_string())

df.to_csv("results.csv", index=False)