from __future__ import annotations

from dataclasses import asdict, field
from datetime import datetime
from typing import Any, List, Optional

try:
    from pydantic.dataclasses import dataclass
except ImportError:  # pragma: no cover - fallback when pydantic isn't available
    from dataclasses import dataclass


@dataclass(slots=True)
class SentimentBreakdown:
    pos: float
    neu: float
    neg: float
    confidence: float
    score: float

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "SentimentBreakdown":
        return cls(**payload)

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Article:
    ticker: str
    title: str
    summary: str
    content: str
    url: str
    source: str
    published_at: datetime
    text: str
    sentiment: Optional[SentimentBreakdown] = None

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "Article":
        data = dict(payload)
        published_at = data.get("published_at")
        if isinstance(published_at, str):
            data["published_at"] = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        sentiment = data.get("sentiment")
        if isinstance(sentiment, dict):
            data["sentiment"] = SentimentBreakdown.model_validate(sentiment)
        return cls(**data)

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["published_at"] = self.published_at.isoformat()
        if self.sentiment is not None:
            payload["sentiment"] = self.sentiment.model_dump(mode=mode)
        return payload


@dataclass(slots=True)
class TickerSentimentReport:
    ticker: str
    window_days: int
    aggregate_score: float
    articles: List[Article] = field(default_factory=list)

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "TickerSentimentReport":
        data = dict(payload)
        data["articles"] = [Article.model_validate(x) if isinstance(x, dict) else x for x in data.get("articles", [])]
        return cls(**data)

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "window_days": self.window_days,
            "aggregate_score": self.aggregate_score,
            "articles": [article.model_dump(mode=mode) for article in self.articles],
        }
