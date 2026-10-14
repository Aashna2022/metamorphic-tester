from transformers import pipeline


class SentimentModel:
    """Wraps a Hugging Face sentiment model behind one simple method."""

    def __init__(self, name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.name = name
        self.pipe = pipeline("sentiment-analysis", model=name)

    def predict(self, text: str) -> str:
        """Return 'POSITIVE' or 'NEGATIVE' for a sentence."""
        return self.pipe(text, truncation=True)[0]["label"]