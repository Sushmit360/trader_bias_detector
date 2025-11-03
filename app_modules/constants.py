SAMPLE_DATASETS = {
    "Sample Pack A": {
        "description": "Balanced mix of low-to-medium risk traders.",
        "trades": "data/examples/trades_sample_a.csv",
        "daily": "data/examples/daily_sample_a.csv",
    },
    "Sample Pack B": {
        "description": "Broader population with varied behaviours.",
        "trades": "data/examples/trades_sample_b.csv",
        "daily": "data/examples/daily_sample_b.csv",
    },
    "Sample Pack C": {
        "description": "Extreme-risk cohort showcasing elevated behavioural flags.",
        "trades": "data/examples/trades_sample_c.csv",
        "daily": "data/examples/daily_sample_c.csv",
    },
}

AI_PROMPT = """
[TARGET_RISK_TIER] = Low/Medium/High

You are a quantitative data generator helping a behavioral risk-scoring team prototype their pipeline.

Goal:
Create two CSV files that capture synthetic equity trades and daily portfolio valuations for at least 12 unique traders. The data will be used to detect overtrading, loss aversion, and herding biases.

Optional control:
[TARGET_RISK_TIER] ∈ {Low, Medium, High, ""}. If provided, tilt behaviors accordingly (e.g., High = more trades, clustered timestamps, wider size dispersion; Low = steadier cadence and sizes).

Output format (very important):
1) Prefer attaching two downloadable files named exactly:
   - trades.csv
   - daily.csv
2) If file attachments are not supported, output two CSV code blocks in this exact order and with these exact fence labels:

-- trades.csv --
```csv
...rows...
```

-- daily.csv --
```csv
...rows...
```

Do not include any extra commentary outside the code fences.

Dataset size (lightweight so most AIs can handle):
- trades.csv: at least 150 rows total (aim 150–200), with a mix of buys and sells for every trader.
- daily.csv: one row per trader per trading day; at least 10 trading days aligned to trades.csv.
- at least 12 traders appear in both files.

File 1: trades.csv
Columns (exact headers):
trader_id,trade_id,timestamp,side,ticker,qty,price,fee,date

Requirements:
- timestamp: ISO-8601 (e.g., 2025-06-10T14:23:05Z) and chronologically valid within each day.
- side: B or S.
- ticker: use a realistic equity ticker universe (e.g., AAPL, MSFT, AMZN, NVDA, TSLA, GOOGL, META, AMD, NFLX) with some concentration on a few names to allow herding detection.
- qty, price, fee: decimals only (no currency symbols). Fees should be small and positive.
- date: YYYY-MM-DD trading date extracted from timestamp.
- trade_id must be unique across the file.
- Behavioral variation:
  - Make each trader’s activity pattern distinct (frequency, time-of-day clustering, tickers, sizes).
  - Include streaky selling after losses (loss aversion) for some traders.
  - Include bursts of many small trades (overtrading) for others.
  - Include synchronized trades (same tickers/times) across multiple traders on select days (herding).
- Time span: spread trades over at least 10 trading days.

File 2: daily.csv
Columns (exact headers):
trader_id,date,portfolio_value

Requirements:
- Provide one row per trader per trading day; dates must align with trades.csv.
- portfolio_value: decimals only; simulate realistic equity curve paths with drawdowns and recoveries so max drawdown can be computed.
- Reflect [TARGET_RISK_TIER] if provided (e.g., High → bigger swings; Low → smoother equity line).

General rules:
- Keep all numeric fields as decimals (no commas or symbols).
- Use consistent trader_id keys across both files (e.g., T01…T12+).
- Ensure valid dates and chronological order within each trader’s timeline.
- No PII and no real customer data; this is synthetic.

Final deliverable order (strict):
1) -- trades.csv --
2) -- daily.csv --
No extra commentary outside the code fences.
"""
