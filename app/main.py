from __future__ import annotations

import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Sequence

from app.config import DEFAULT_CONFIG, AppConfig
from app.models.schemas import Article, TickerSentimentReport
from app.nlp.scoring import aggregate_ticker_sentiment, deduplicate_articles
from app.sources.alphavantage import AlphaVantageClient
from app.sources.newsapi import NewsApiClient
from app.storage.cache import DiskCache


def get_source_client(source: str, config: AppConfig, cache: DiskCache):
    if source == "alphavantage":
        return AlphaVantageClient(config, cache)
    if source == "newsapi":
        return NewsApiClient(config, cache)
    raise ValueError(f"Unsupported source '{source}'")


def fetch_articles(
    tickers: Sequence[str],
    days: int,
    source: str,
    config: AppConfig = DEFAULT_CONFIG,
) -> list[Article]:
    cache = DiskCache(config.cache_dir, ttl_seconds=config.cache_ttl_seconds)
    client = get_source_client(source, config, cache)
    articles = client.fetch(list(tickers), days)
    return deduplicate_articles(articles)


def score_tickers(
    tickers: Sequence[str],
    days: int,
    source: str,
    export_path: str | None = None,
    config: AppConfig = DEFAULT_CONFIG,
) -> list[TickerSentimentReport]:
    cache = DiskCache(config.cache_dir, ttl_seconds=config.cache_ttl_seconds)
    fetch_key = f"scored:{','.join(sorted(tickers))}:{days}:{source}"
    cached = cache.get(fetch_key)
    if cached:
        reports = [TickerSentimentReport.model_validate(x) for x in cached]
        if export_path:
            _write_export(export_path, reports)
        return reports

    articles = fetch_articles(tickers, days, source, config=config)
    if articles:
        try:
            from app.nlp.finbert import FinBertSentimentAnalyzer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT dependencies missing. Install requirements (torch, transformers) to run scoring."
            ) from exc

        analyzer = FinBertSentimentAnalyzer()
        sentiments = analyzer.predict([a.text for a in articles])
        for article, sentiment in zip(articles, sentiments):
            article.sentiment = sentiment

    by_ticker: dict[str, list[Article]] = {ticker: [] for ticker in tickers}
    for article in articles:
        by_ticker.setdefault(article.ticker, []).append(article)

    reports: list[TickerSentimentReport] = []
    for ticker in tickers:
        ticker_articles = by_ticker.get(ticker, [])
        aggregate = aggregate_ticker_sentiment(
            ticker_articles,
            source_weights=config.source_credibility_weights,
        )
        reports.append(
            TickerSentimentReport(
                ticker=ticker,
                window_days=days,
                aggregate_score=aggregate,
                articles=ticker_articles,
            )
        )

    cache.set(fetch_key, [r.model_dump(mode="json") for r in reports])
    if export_path:
        _write_export(export_path, reports)
    return reports


def _write_export(path: str, reports: list[TickerSentimentReport]) -> None:
    payload = [r.model_dump(mode="json") for r in reports]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock news sentiment analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("fetch", "score"):
        p = subparsers.add_parser(name)
        p.add_argument("--tickers", nargs="+", required=True)
        p.add_argument("--days", type=int, default=7)
        p.add_argument("--source", choices=["alphavantage", "newsapi"], default="alphavantage")
        if name == "score":
            p.add_argument("--export", type=str, default=None)

    api = subparsers.add_parser("serve")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--source", choices=["alphavantage", "newsapi"], default="alphavantage")
    return parser


def _print_articles(articles: list[Article]) -> None:
    payload = []
    for item in articles:
        payload.append(
            {
                "ticker": item.ticker,
                "title": item.title,
                "url": str(item.url),
                "source": item.source,
                "published_at": item.published_at.astimezone(timezone.utc).isoformat(),
                "summary": item.summary,
            }
        )
    print(json.dumps(payload, indent=2))


def _print_reports(reports: list[TickerSentimentReport]) -> None:
    print(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            articles = fetch_articles(args.tickers, args.days, args.source)
            _print_articles(articles)
            return 0

        if args.command == "score":
            reports = score_tickers(args.tickers, args.days, args.source, args.export)
            _print_reports(reports)
            return 0

        if args.command == "serve":
            run_api(args.host, args.port, args.source)
            return 0
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    return 1


def run_api(host: str, port: int, source: str) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, Query
    except ImportError as exc:
        raise RuntimeError("FastAPI server requested but fastapi/uvicorn not installed") from exc

    app = FastAPI(title="Stock News Sentiment API")

    @app.get("/sentiment")
    def sentiment(ticker: str = Query(...), days: int = Query(7, ge=1, le=30)):
        reports = score_tickers([ticker], days=days, source=source)
        return reports[0].model_dump(mode="json")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    raise SystemExit(run_cli())
