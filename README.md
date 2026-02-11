# TweetSentimentAnalysis

Python NLP application to fetch stock-related news and compute per-article and aggregate sentiment per ticker.

## Features

- Fetches stock news from:
  - **Alpha Vantage** (`NEWS_SENTIMENT`) using `ALPHAVANTAGE_API_KEY`.
  - **NewsAPI** (`everything`) using `NEWSAPI_KEY`.
- Normalized article schema with UTC timestamps.
- Financial sentiment using **FinBERT** (`ProsusAI/finbert`).
- Per-article sentiment probabilities + scalar score:
  - `score = (P(positive) - P(negative)) * confidence`
  - `confidence = max(P(pos), P(neu), P(neg))`
- Aggregate ticker sentiment weighted by:
  - recency (exponential decay, half-life = 3 days)
  - optional source credibility weights.
- Caching for raw API responses and scored results (`./cache`, default TTL 30m).
- Retries + exponential backoff for API calls.
- Deduplication by URL + title hash.
- CLI plus optional FastAPI endpoint.

## Project Structure

```text
app/
  __init__.py
  __main__.py
  main.py
  config.py
  sources/
    alphavantage.py
    newsapi.py
  nlp/
    finbert.py
    scoring.py
  storage/
    cache.py
  models/
    schemas.py
tests/
  test_scoring.py
  test_dedup.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Set at least one source key based on selected source:

```bash
export ALPHAVANTAGE_API_KEY="..."
export NEWSAPI_KEY="..."
```

## CLI Usage

Fetch normalized news:

```bash
python -m app fetch --tickers AAPL TSLA --days 7 --source alphavantage
```

Score sentiment and export:

```bash
python -m app score --tickers AAPL --days 7 --source alphavantage --export out.json
```

## Optional API

```bash
python -m app serve --source alphavantage --host 0.0.0.0 --port 8000
```

Then request:

```bash
curl "http://localhost:8000/sentiment?ticker=AAPL&days=7"
```

## Example Output

```json
{
  "ticker": "AAPL",
  "window_days": 7,
  "aggregate_score": 0.23,
  "articles": [
    {
      "title": "Apple posts stronger services growth",
      "url": "https://example.com/apple-news",
      "published_at": "2026-01-01T12:00:00+00:00",
      "sentiment": {
        "pos": 0.62,
        "neu": 0.30,
        "neg": 0.08,
        "confidence": 0.62,
        "score": 0.3348
      }
    }
  ]
}
```

## Testing

```bash
pytest -q
```

## Notes

- API keys are never hard-coded.
- Missing key errors are explicit and tied to source selection.
- First FinBERT run downloads model artifacts from Hugging Face.
