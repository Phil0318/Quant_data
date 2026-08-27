import os
import sys
import datetime as dt

import pandas as pd
import yfinance as yf

TICKERS = ["MSFT", "GOOG"]
MAX_EXPIRIES = 10
OUT = "data/options_snapshots.csv"


def snapshot(ticker, max_expiries=MAX_EXPIRIES):
    t = yf.Ticker(ticker)
    exps = list(t.options)[:max_expiries]
    hist = t.history(period="1d", auto_adjust=True)
    underlying = float(hist["Close"].iloc[-1])
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    rows = []
    for exp in exps:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue
        for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
            for _, r in df.iterrows():
                rows.append({
                    "snapshot_time": now,
                    "ticker": ticker,
                    "expiration": exp,
                    "option_type": opt_type,
                    "strike": r["strike"],
                    "last": r["lastPrice"],
                    "bid": r["bid"],
                    "ask": r["ask"],
                    "iv": r["impliedVolatility"],
                    "volume": r["volume"],
                    "open_interest": r["openInterest"],
                    "in_the_money": r["inTheMoney"],
                    "underlying": underlying,
                })
    return pd.DataFrame(rows)


def main():
    frames = []
    for t in TICKERS:
        try:
            df = snapshot(t)
            frames.append(df)
            print(f"{t}: {len(df)} 行", flush=True)
        except Exception as e:
            print(f"{t} 失败: {e}", flush=True)

    if not frames:
        print("无有效数据，退出", flush=True)
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    header = not os.path.exists(OUT)
    out.to_csv(OUT, mode="a", header=header, index=False)
    print(f"已追加 {len(out)} 行到 {OUT}", flush=True)


if __name__ == "__main__":
    main()