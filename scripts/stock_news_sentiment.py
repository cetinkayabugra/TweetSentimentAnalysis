"""Fetch stock news, score sentiment, and render an interactive HTML report.

Usage:
    python scripts/stock_news_sentiment.py --ticker AAPL --limit 20 --output reports/aapl_news_sentiment.html
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve stock news from Yahoo Finance RSS and render sentiment analysis to HTML."
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument(
        "--limit", type=int, default=25, help="Maximum number of news items to analyze (default: 25)"
    )
    parser.add_argument(
        "--output",
        default="reports/stock_news_sentiment.html",
        help="Output HTML file path (default: reports/stock_news_sentiment.html)",
    )
    return parser.parse_args()


def fetch_rss_news(ticker: str, limit: int) -> list[dict[str, str]]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    items: list[dict[str, str]] = []

    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        if title:
            items.append(
                {
                    "title": title,
                    "description": description,
                    "link": link,
                    "published_at": pub_date,
                    "content": f"{title}. {description}".strip(),
                }
            )

    return items


def ensure_vader_available() -> None:
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    ensure_vader_available()

    from nltk.sentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    polarity = df["content"].apply(analyzer.polarity_scores).apply(pd.Series)

    scored = pd.concat([df, polarity], axis=1)
    scored["sentiment_label"] = pd.cut(
        scored["compound"],
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["Negative", "Neutral", "Positive"],
    )

    scored["published_at"] = pd.to_datetime(scored["published_at"], errors="coerce", utc=True)
    scored = scored.sort_values("published_at", ascending=False, na_position="last")
    return scored


def build_dashboard(df: pd.DataFrame, ticker: str) -> str:
    color_map = {"Negative": "#D62728", "Neutral": "#7F7F7F", "Positive": "#2CA02C"}

    fig = px.bar(
        df,
        x="published_at",
        y="compound",
        color="sentiment_label",
        hover_data=["title", "description", "link"],
        color_discrete_map=color_map,
        title=f"{ticker.upper()} News Sentiment (VADER Compound Score)",
    )

    fig.update_layout(
        xaxis_title="Published time",
        yaxis_title="Compound sentiment score",
        template="plotly_white",
        legend_title="Sentiment",
    )

    summary = df["sentiment_label"].value_counts(dropna=False).rename_axis("sentiment").reset_index(name="count")
    summary_table = summary.to_html(index=False)

    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return f"""
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <title>{ticker.upper()} News Sentiment</title>
      </head>
      <body style=\"font-family: Arial, sans-serif; max-width: 1200px; margin: 2rem auto;\">
        <h1>{ticker.upper()} News Sentiment Dashboard</h1>
        <p>Generated from Yahoo Finance RSS feed and scored with NLTK VADER.</p>
        <h2>Sentiment Summary</h2>
        {summary_table}
        <h2>Sentiment by article</h2>
        {fig_html}
      </body>
    </html>
    """


def main() -> int:
    args = parse_args()

    try:
        news = fetch_rss_news(args.ticker, args.limit)
        if not news:
            print(f"No news items found for ticker {args.ticker}.")
            return 1

        raw_df = pd.DataFrame(news)
        scored_df = score_sentiment(raw_df)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(build_dashboard(scored_df, args.ticker), encoding="utf-8")

        csv_path = output_path.with_suffix(".csv")
        scored_df.to_csv(csv_path, index=False)

        print(f"HTML dashboard written to: {output_path}")
        print(f"Scored data written to: {csv_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
