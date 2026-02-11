from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, List

from app.config import AppConfig
from app.models.schemas import Article
from app.storage.cache import DiskCache
from app.sources.http_utils import get_client

NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsApiClient:
    """Fetches and normalizes stock news from NewsAPI."""

    def __init__(self, config: AppConfig, cache: DiskCache) -> None:
        if not config.newsapi_key:
            raise ValueError("NEWSAPI_KEY is required when source=newsapi")
        self.config = config
        self.cache = cache
        self.http = get_client()

    def fetch(self, tickers: list[str], days: int, company_names: dict[str, str] | None = None) -> list[Article]:
        company_names = company_names or {}
        all_articles: list[Article] = []
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        for ticker in tickers:
            query = ticker
            if company_names.get(ticker):
                query = f"{ticker} OR \"{company_names[ticker]}\""
            key = f"newsapi:{ticker}:{days}:{query}"
            cached = self.cache.get(key)
            if cached:
                all_articles.extend(Article.model_validate(item) for item in cached)
                continue
            params = {
                "q": query,
                "from": since,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": self.config.newsapi_key,
            }
            payload = self._request_with_retries(params)
            items = self._normalize(payload, ticker)
            self.cache.set(key, [item.model_dump(mode="json") for item in items])
            all_articles.extend(items)
        return all_articles

    def _request_with_retries(self, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self.http.get(NEWSAPI_URL, params=params, timeout=self.config.request_timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        return data
            except Exception as exc:  # network/provider failure
                last_error = exc
            sleep = self.config.backoff_base_seconds * (2**attempt)
            time.sleep(sleep)

        if last_error:
            raise RuntimeError(f"NewsAPI request failed after retries: {last_error}") from last_error
        raise RuntimeError("NewsAPI request failed after retries")

    def _normalize(self, payload: dict[str, Any], ticker: str) -> list[Article]:
        articles: List[Article] = []
        for item in payload.get("articles", []):
            title = (item.get("title") or "").strip()
            summary = (item.get("description") or "").strip()
            content = (item.get("content") or "").strip()
            url = item.get("url")
            source = ((item.get("source") or {}).get("name") or "").strip()
            published = self._parse_timestamp(item.get("publishedAt", ""))
            if not title or not url or not published:
                continue
            text = _clean_text(" ".join([title, summary, content]))
            if not text:
                continue
            articles.append(
                Article(
                    ticker=ticker,
                    title=title,
                    summary=summary,
                    content=content,
                    url=url,
                    source=source,
                    published_at=published,
                    text=text,
                )
            )
        return articles

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime | None:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None


def _clean_text(text: str) -> str:
    import re

    no_html = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", no_html).strip()
    return cleaned
