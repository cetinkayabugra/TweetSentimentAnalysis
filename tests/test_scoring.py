from datetime import datetime, timedelta, timezone

from app.models.schemas import Article, SentimentBreakdown
from app.nlp.scoring import aggregate_ticker_sentiment, sentiment_scalar


def make_article(hours_ago: int, score: float, source: str = "Reuters") -> Article:
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return Article(
        ticker="AAPL",
        title=f"Title {hours_ago}",
        summary="Summary",
        content="Content",
        url=f"https://example.com/{hours_ago}",
        source=source,
        published_at=ts,
        text="Title Summary",
        sentiment=SentimentBreakdown(pos=0.5, neu=0.4, neg=0.1, confidence=0.5, score=score),
    )


def test_sentiment_scalar_formula():
    confidence, score = sentiment_scalar(pos=0.62, neu=0.30, neg=0.08)
    assert round(confidence, 2) == 0.62
    assert round(score, 4) == round((0.62 - 0.08) * 0.62, 4)


def test_aggregation_weights_recency_and_source():
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    recent = make_article(hours_ago=1, score=0.7, source="Reuters")
    older = make_article(hours_ago=72, score=-0.6, source="Unknown")
    recent.published_at = now - timedelta(hours=1)
    older.published_at = now - timedelta(hours=72)

    score = aggregate_ticker_sentiment(
        [recent, older],
        source_weights={"reuters": 1.2},
        half_life_days=3,
        now=now,
    )

    assert score > 0
    assert score < 0.7
