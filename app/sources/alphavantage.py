from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, List

from app.config import AppConfig
from app.models.schemas import Article
from app.storage.cache import DiskCache
from app.sources.http_utils import get_client

ALPHA_URL = "https://www.alphavantage.co/query"


class AlphaVantageClient:
    """Fetches and normalizes stock news from Alpha Vantage."""

    def __init__(self, config: AppConfig, cache: DiskCache) -> None:
        if not config.alpha_vantage_api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is required when source=alphavantage")
        self.config = config
        self.cache = cache
        self.http = get_client()

    def fetch(self, tickers: list[str], days: int) -> list[Article]:
        all_articles: list[Article] = []
        for ticker in tickers:
            key = f"alphavantage:{ticker}:{days}"
            cached = self.cache.get(key)
            if cached:
                all_articles.extend(Article.model_validate(item) for item in cached)
                continue

            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 200,
                "apikey": self.config.alpha_vantage_api_key,
            }
            payload = self._request_with_retries(params)
            items = self._normalize(payload, ticker=ticker, days=days)
            self.cache.set(key, [item.model_dump(mode="json") for item in items])
            all_articles.extend(items)
        return all_articles

    def _request_with_retries(self, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self.http.get(ALPHA_URL, params=params, timeout=self.config.request_timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    if "Note" not in data:
                        return data
            except Exception as exc:  # network/provider failure
                last_error = exc
            sleep = self.config.backoff_base_seconds * (2**attempt)
            time.sleep(sleep)

        if last_error:
            raise RuntimeError(f"Alpha Vantage request failed after retries: {last_error}") from last_error
        raise RuntimeError("Alpha Vantage request failed after retries")

    def _normalize(self, payload: dict[str, Any], ticker: str, days: int) -> list[Article]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        articles: List[Article] = []
        for item in payload.get("feed", []):
            published_raw = item.get("time_published", "")
            published = self._parse_alpha_timestamp(published_raw)
            if not published or published < cutoff:
                continue
            title = (item.get("title") or "").strip()
            summary = (item.get("summary") or "").strip()
            url = item.get("url")
            if not title or not url:
                continue
            text = _clean_text(" ".join([title, summary]))
            if not text:
                continue
            source = (item.get("source") or "").strip()
            articles.append(
                Article(
                    ticker=ticker,
                    title=title,
                    summary=summary,
                    content=item.get("summary") or "",
                    url=url,
                    source=source,
                    published_at=published,
                    text=text,
                )
            )
        return articles

    @staticmethod
    def _parse_alpha_timestamp(raw: str) -> datetime | None:
        try:
            # Example: 20240101T153000
            dt = datetime.strptime(raw, "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _clean_text(text: str) -> str:
    import re

    no_html = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", no_html).strip()
    return cleaned
