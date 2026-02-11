from datetime import datetime, timedelta, timezone

from app.models.schemas import Article
from app.nlp.scoring import deduplicate_articles


def build_article(title: str, url: str, minutes_ago: int) -> Article:
    return Article(
        ticker="TSLA",
        title=title,
        summary="summary",
        content="",
        url=url,
        source="Source",
        published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        text=title,
    )


def test_deduplicate_articles_by_url_and_title_hash():
    a1 = build_article("Tesla rallies", "https://example.com/news/1", 2)
    a2 = build_article("Tesla rallies", "https://example.com/news/1", 1)
    a3 = build_article("Tesla dips", "https://example.com/news/2", 1)

    deduped = deduplicate_articles([a1, a2, a3])

    assert len(deduped) == 2
    urls = {str(a.url) for a in deduped}
    assert "https://example.com/news/1" in urls
    assert "https://example.com/news/2" in urls


def test_deduplicate_keeps_most_recent_duplicate():
    older = build_article("Same title", "https://example.com/dup", 5)
    newer = build_article("Same title", "https://example.com/dup", 1)

    deduped = deduplicate_articles([older, newer])

    assert len(deduped) == 1
    assert deduped[0].published_at == newer.published_at
