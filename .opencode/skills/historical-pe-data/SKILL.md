---
name: historical-pe-data
description: Use when backtesting P/E (price-to-earnings) or valuation-based strategies on US stocks and need long historical P/E history. Covers the yfinance limitation and the Macrotrends scraping workaround, including rate-limit handling. Trigger keywords: historical P/E, PE ratio history, valuation data, P/E backtest, pe-ratio.
---

# Getting Historical P/E Data

## The problem: yfinance gives almost no earnings history

yfinance can compute *current* P/E via `Ticker.info["trailingPE"]`, but its
fundamental endpoints only return the **most recent ~5 quarters**:

- `Ticker.quarterly_financials` / `Ticker.quarterly_income_stmt` → 5 columns
- `Ticker.income_stmt` (annual) → 5 years
- `Ticker.earnings_history` → 4 quarters of `epsActual`

None of these are enough to compute a rolling percentile of P/E over 5-10
years. Do **not** try to reconstruct a long P/E series from yfinance.

## The solution: scrape Macrotrends (free, no API key)

Macrotrends hosts quarterly P/E history (~80 quarters, back to ~2006) as a
plain HTML table. URL pattern:

```
https://www.macrotrends.net/stocks/charts/{TICKER}/{SLUG}/pe-ratio
```

- `{TICKER}` = uppercase symbol, e.g. `AAPL`
- `{SLUG}` = lowercase company slug, e.g. `apple`, `microsoft`, `alphabet`,
  `amazon`, `nvidia`, `meta-platforms`. If the slug is wrong the page 404s;
  the ticker part is what matters most but the slug should match the company
  name.

### Working implementation (in `pe_strategy.py`)

```python
import urllib.request, re, html as h, pandas as pd

def get_pe_history(ticker, name):
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{name}/pe-ratio"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
    tables = re.findall(r"<table.*?</table>", page, re.DOTALL)
    rows = re.findall(r"<tr.*?</tr>", tables[0], re.DOTALL)  # FIRST table = P/E history
    data = []
    for r in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.DOTALL)
        clean = [h.unescape(c).strip() for c in cells]
        if len(clean) >= 4 and clean[0]:
            try:
                data.append({"date": pd.to_datetime(clean[0]),
                             "price": float(clean[1]), "pe": float(clean[3])})
            except (ValueError, TypeError):
                pass
    return pd.DataFrame(data).sort_values("date").set_index("date")
```

Key parsing facts:

- The **first `<table>`** on the page is the quarterly P/E table (there are 3
  tables: P/E history, company description, peer comparison).
- Each data row's `<td>` cells are `[date, price, ?, pe]` — column index 0 is
  the date, index 1 is price, index 3 is the P/E ratio. (Column 2 is empty in
  the P/E table.)
- Values are HTML-escaped; run them through `html.unescape` and strip.

## Rate limiting (important)

Macrotrends returns **HTTP 429 Too Many Requests** when hit in rapid
succession. When looping over multiple tickers:

```python
time.sleep(2)  # or 3 seconds between requests
```

On a 429, sleep ~5s and retry once. Without this, later tickers in a loop
will silently fail.

## Combining with price for the strategy

P/E comes in quarterly (quarter-end) rows. To backtest daily:

```python
px = yf.Ticker(t).history(period="10y", auto_adjust=True)["Close"]
px.index = px.index.tz_localize(None)          # drop tz for clean alignment
pe_daily = pe["pe"].reindex(px.index, method="ffill").bfill()  # ffill quarterly -> daily
```

Then compute a rolling percentile (e.g. 5-year = 20 quarters) of the P/E and
threshold on it. See `pe_strategy.py::pe_strategy` (state machine) and
`pe_strategy.py::pe_linear_strategy` (continuous position sizing).

## Gotchas

- **The rank/percentile edge case**: `rolling(...).rank(pct=True)` can land
  exactly on the threshold (e.g. 25.0). Use `<=`/`>=` for buy/sell, not
  strict `<`/`>`, or the strategy may never trigger.
- **`fillna(method=...)` is deprecated** in pandas 2.x — use `.ffill()` /
  `.bfill()` instead.
- P/E data is **trailing twelve months (TTM)** as published by Macrotrends;
  it can go negative for loss-making companies, which breaks percentile logic.
  Filter or cap negative/zero P/E before ranking.
- The strategy results are environment-dependent: P/E timing works in
  range-bound/choppy markets but consistently underperforms buy-and-hold on
  growth megacaps (AAPL, NVDA) in long bull runs.

## Reference files in this repo

- `pe_strategy.py` — `get_pe_history()` (scraping), `pe_strategy()` (state
  machine), `pe_linear_strategy()` (continuous sizing), `performance()`.
- `plot_pe_result.py` — 2x3 subplot of P/E + buy/sell markers + equity curves.
