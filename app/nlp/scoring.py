from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone

from app.models.schemas import Article


def sentiment_scalar(pos: float, neu: float, neg: float) -> tuple[float, float]:
    """Compute confidence and scalar sentiment score."""
    confidence = max(pos, neu, neg)
    score = (pos - neg) * confidence
    return confidence, score


def deduplicate_articles(articles: list[Article]) -> list[Article]:
    """Deduplicate articles by URL and title hash, keeping the most recent first."""
    sorted_articles = sorted(articles, key=lambda a: a.published_at, reverse=True)
    seen: set[str] = set()
    deduped: list[Article] = []
    for article in sorted_articles:
        key = _article_key(article)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def _article_key(article: Article) -> str:
    title_hash = hashlib.sha256(article.title.lower().strip().encode("utf-8")).hexdigest()
    return f"{article.url}|{title_hash}"


def aggregate_ticker_sentiment(
    articles: list[Article],
    source_weights: dict[str, float] | None = None,
    half_life_days: float = 3.0,
    now: datetime | None = None,
) -> float:
    """Aggregate sentiment with recency and source weighting."""
    if not articles:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)

    source_weights = source_weights or {}
    numerator = 0.0
    denominator = 0.0
    for article in articles:
        if article.sentiment is None:
            continue
        age_days = max((now - article.published_at).total_seconds() / 86400.0, 0.0)
        recency_weight = math.exp(-math.log(2) * age_days / half_life_days)
        source_key = article.source.lower().strip()
        credibility_weight = source_weights.get(source_key, 1.0)
        weight = recency_weight * credibility_weight

        # Keep neutral-heavy samples stable by naturally shrinking with the score formula.
        numerator += article.sentiment.score * weight
        denominator += weight

    if denominator == 0:
        return 0.0
    return numerator / denominator
