# Scripts

Command-line utilities and one-off scripts for data collection and experiments.

## Stock news sentiment dashboard

Use `stock_news_sentiment.py` to:

1. Retrieve stock-related news headlines from Yahoo Finance RSS.
2. Score each article with NLTK VADER sentiment.
3. Generate an interactive HTML dashboard (Plotly) and a scored CSV file.

Example:

```bash
python scripts/stock_news_sentiment.py --ticker AAPL --limit 25 --output reports/aapl_news_sentiment.html
```

Dependencies: `pandas`, `requests`, `nltk`, `plotly`.
