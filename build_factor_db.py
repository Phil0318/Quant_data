import urllib.request
import re
import html as h
import time
import os
import sys

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
SLEEP = 4
OUT_CSV = "factor_db.csv"
FAILED_CSV = "factor_failed.csv"


def log(msg):
    print(msg, flush=True)

FEATURES = [
    "pe_level", "pe_pctile", "pe_z",
    "eps_yoy", "eps_accel", "eps_qoq",
    "mom_3m", "mom_6m", "mom_12m", "ma_dev",
]


def _read_html(url):
    req = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
    return pd.read_html(html)


def get_constituents():
    sp = _read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    sp500 = sp[["Symbol", "Security"]].rename(
        columns={"Symbol": "ticker", "Security": "name"})

    nasdaq = _read_html("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies")[0]
    ndx100 = nasdaq[["Ticker", "Company"]].rename(
        columns={"Ticker": "ticker", "Company": "name"})

    both = pd.concat([sp500, ndx100], ignore_index=True)
    both = both.drop_duplicates(subset="ticker").reset_index(drop=True)
    return both


def slugify(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    return name.strip("-")


def fetch_pe(ticker, slug, retries=3):
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{slug}/pe-ratio"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
            tables = re.findall(r"<table.*?</table>", page, re.DOTALL)
            rows = re.findall(r"<tr.*?</tr>", tables[0], re.DOTALL)
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
            df = pd.DataFrame(data)
            if df.empty:
                return None
            return df.sort_values("date").set_index("date")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 120 * (attempt + 1)
                log(f"   429限流, 等待{wait}s (第{attempt+1}次重试)...")
                time.sleep(wait)
            else:
                raise
        except Exception:
            wait = 20 * (attempt + 1)
            log(f"   网络错误, 等待{wait}s (第{attempt+1}次重试)...")
            time.sleep(wait)
    raise RuntimeError("重试耗尽")


def compute_factors(pe):
    pe = pe.sort_index()
    eps = pe["price"] / pe["pe"]
    price_q = pe["price"]

    pe_level = pe["pe"]
    pe_pctile = pe["pe"].rolling(20, min_periods=20).rank(pct=True) * 100
    pe_z = (pe["pe"] - pe["pe"].rolling(20).mean()) / pe["pe"].rolling(20).std()

    eps_yoy = eps / eps.shift(4) - 1
    eps_accel = eps_yoy - eps_yoy.shift(4)
    eps_qoq = eps / eps.shift(1) - 1

    mom_3m = price_q / price_q.shift(1) - 1
    mom_6m = price_q / price_q.shift(2) - 1
    mom_12m = price_q / price_q.shift(4) - 1
    ma_dev = price_q / price_q.rolling(20).mean() - 1

    label = price_q.shift(-1) / price_q - 1

    out = pd.DataFrame({
        "pe_level": pe_level, "pe_pctile": pe_pctile, "pe_z": pe_z,
        "eps_yoy": eps_yoy, "eps_accel": eps_accel, "eps_qoq": eps_qoq,
        "mom_3m": mom_3m, "mom_6m": mom_6m, "mom_12m": mom_12m, "ma_dev": ma_dev,
        "label": label,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    ratio_cols = ["pe_z", "eps_yoy", "eps_accel", "eps_qoq",
                  "mom_3m", "mom_6m", "mom_12m", "ma_dev"]
    out[ratio_cols] = out[ratio_cols].clip(-5, 5)
    out["label"] = out["label"].clip(-5, 5)
    return out


def load_done():
    if os.path.exists(OUT_CSV):
        try:
            existing = pd.read_csv(OUT_CSV, usecols=["ticker"])
            return set(existing["ticker"].unique())
        except Exception:
            return set()
    return set()


def main():
    constituents = get_constituents()
    log(f"成分股总数: {len(constituents)} (S&P500 + NASDAQ100 去重)")

    done = load_done()
    if done:
        log(f"已完成 {len(done)} 只，跳过续传")
    todo = constituents[~constituents["ticker"].isin(done)]
    log(f"待处理: {len(todo)} 只\n")

    for idx, row in todo.iterrows():
        ticker = row["ticker"]
        slug = slugify(row["name"])
        try:
            pe = fetch_pe(ticker, slug)
            if pe is None or len(pe) < 20:
                log(f"[{idx+1}/{len(todo)}] {ticker}: 无足够数据，跳过")
                pd.DataFrame([[ticker, row["name"], "no_data"]]).to_csv(
                    FAILED_CSV, mode="a", header=False, index=False)
                time.sleep(SLEEP)
                continue

            factors = compute_factors(pe)
            factors["ticker"] = ticker
            header = not os.path.exists(OUT_CSV)
            factors.to_csv(OUT_CSV, mode="a", header=header, index_label="date")
            log(f"[{idx+1}/{len(todo)}] {ticker}: OK ({len(factors)} 季度)")
        except Exception as e:
            pd.DataFrame([[ticker, row["name"], str(e)[:50]]]).to_csv(
                FAILED_CSV, mode="a", header=False, index=False)
            log(f"[{idx+1}/{len(todo)}] {ticker}: 失败 {str(e)[:40]}")
        time.sleep(SLEEP)

    print(f"\n完成。数据存于 {OUT_CSV}，失败列表 {FAILED_CSV}", flush=True)

if __name__ == "__main__":
    main()