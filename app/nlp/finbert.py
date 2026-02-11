from __future__ import annotations

from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.schemas import SentimentBreakdown
from app.nlp.scoring import sentiment_scalar


class FinBertSentimentAnalyzer:
    """Financial sentiment classifier using ProsusAI/finbert."""

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 16) -> None:
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def predict(self, texts: Iterable[str]) -> list[SentimentBreakdown]:
        text_list = list(texts)
        outputs: list[SentimentBreakdown] = []
        with torch.no_grad():
            for i in range(0, len(text_list), self.batch_size):
                batch = text_list[i : i + self.batch_size]
                tokens = self.tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
                logits = self.model(**tokens).logits
                probs = torch.softmax(logits, dim=1).tolist()
                for prob in probs:
                    pos = float(prob[0])
                    neg = float(prob[1])
                    neu = float(prob[2])
                    confidence, score = sentiment_scalar(pos=pos, neu=neu, neg=neg)
                    outputs.append(
                        SentimentBreakdown(
                            pos=pos,
                            neu=neu,
                            neg=neg,
                            confidence=confidence,
                            score=score,
                        )
                    )
        return outputs
