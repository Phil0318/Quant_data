import urllib.request
import re
import html as h
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti"
]
matplotlib.rcParams["axes.unicode_minus"] = False

TICKERS = [
    ("AAPL", "apple"), ("MSFT", "microsoft"), ("GOOG", "alphabet"),
    ("AMZN", "amazon"), ("NVDA", "nvidia"), ("META", "meta-platforms"),
]
TICKER_NAMES = {"AAPL": "苹果", "MSFT": "微软", "GOOG": "谷歌",
                "AMZN": "亚马逊", "NVDA": "英伟达", "META": "Meta"}


def get_pe_history(ticker, name):
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{name}/pe-ratio"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
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
    return pd.DataFrame(data).sort_values("date").set_index("date")


def pe_strategy(price, pe, lookback_years=5, low_pct=10, high_pct=90, fee=0.001):
    pe = pe.sort_index()
    pe_daily = pe["pe"].reindex(price.index, method="ffill")
    pe_daily = pe_daily.bfill()

    window = lookback_years * 4
    pctile = pe["pe"].rolling(window, min_periods=window).rank(pct=True) * 100
    pctile_daily = pctile.reindex(price.index, method="ffill").ffill().bfill().fillna(50)

    position = pd.Series(0, index=price.index)
    current = 0
    for i in range(len(price)):
        if pctile_daily.iloc[i] <= low_pct:
            current = 1
        elif pctile_daily.iloc[i] >= high_pct:
            current = 0
        position.iloc[i] = current

    position = position.shift(1).fillna(0)
    ret = price.pct_change().fillna(0)
    cost = position.diff().abs().fillna(0) * fee
    strat_ret = position * ret - cost
    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + ret).cumprod()

    return equity, bh_equity, position, pctile_daily


def pe_linear_strategy(price, pe, lookback_years=5, low_pct=25, high_pct=75, fee=0.001):
    pe = pe.sort_index()
    window = lookback_years * 4
    pctile = pe["pe"].rolling(window, min_periods=window).rank(pct=True) * 100
    pctile_daily = pctile.reindex(price.index, method="ffill").ffill().bfill().fillna(50)

    target = (high_pct - pctile_daily) / (high_pct - low_pct)
    target = target.clip(0, 1)

    position = target.shift(1).fillna(0)
    ret = price.pct_change().fillna(0)
    cost = position.diff().abs().fillna(0) * fee
    strat_ret = position * ret - cost
    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + ret).cumprod()

    return equity, bh_equity, position, pctile_daily


def performance(equity):
    if equity.empty or len(equity) < 2:
        return {"总收益": np.nan, "年化收益": np.nan, "夏普比率": np.nan, "最大回撤": np.nan}
    total_ret = equity.iloc[-1] - 1
    ann_ret = equity.iloc[-1] ** (252 / len(equity)) - 1
    daily_ret = equity.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252 - 0.02) / vol if vol > 0 else 0
    peak = equity.cummax()
    max_dd = (equity / peak - 1).min()
    return {"总收益": total_ret, "年化收益": ann_ret,
            "夏普比率": sharpe, "最大回撤": max_dd}


def main():
    print("正在下载 P/E 数据...")
    results = []
    for ticker, name in TICKERS:
        try:
            pe = get_pe_history(ticker, name)
            print(f"  {ticker}: P/E历史 {len(pe)} 个季度 ({pe.index[0].date()}~{pe.index[-1].date()})")
        except Exception as e:
            print(f"  {ticker}: P/E获取失败 - {e}")
            time.sleep(5)
            continue
        time.sleep(3)

        try:
            px = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"]
            px.index = px.index.tz_localize(None)
            px = px[px.index >= pe.index[0]]
        except Exception as e:
            print(f"  {ticker}: 价格获取失败 - {e}")
            continue

        equity, bh_equity, position, pctile = pe_strategy(px, pe)
        results.append({
            "ticker": ticker, "name": TICKER_NAMES[ticker],
            "strat": performance(equity), "bh": performance(bh_equity),
            "in_pct": position.mean(),
            "equity": equity, "bh_equity": bh_equity,
            "pe": pe["pe"], "pctile": pctile, "position": position,
        })

    if not results:
        print("无有效数据")
        return

    print(f"\n{'='*80}")
    print(f"{'股票':<8}{'P/E低位策略':^24}{'买入持有':^22}{'持仓占比':>8}")
    print(f"{'':8}{'总收益':>8}{'年化':>8}{'夏普':>8}{'总收益':>8}{'年化':>8}{'夏普':>8}")
    print(f"{'='*80}")
    for r in results:
        s, b = r["strat"], r["bh"]
        print(f"{r['name']:<8}"
              f"{s['总收益']:>8.1%}{s['年化收益']:>8.1%}{s['夏普比率']:>8.2f}"
              f"{b['总收益']:>8.1%}{b['年化收益']:>8.1%}{b['夏普比率']:>8.2f}"
              f"{r['in_pct']:>8.1%}")

    avg_pe_strat = np.mean([r["strat"]["年化收益"] for r in results])
    avg_pe_sharpe = np.mean([r["strat"]["夏普比率"] for r in results])
    avg_bh_ann = np.mean([r["bh"]["年化收益"] for r in results])
    avg_bh_sharpe = np.mean([r["bh"]["夏普比率"] for r in results])
    print(f"\n平均:  P/E策略 年化 {avg_pe_strat:.1%}  夏普 {avg_pe_sharpe:.2f}"
          f"  |  买入持有 年化 {avg_bh_ann:.1%}  夏普 {avg_bh_sharpe:.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r["pe"].index, r["pe"], linewidth=1.3, color="steelblue",
                label="P/E")
        pos = r["position"]
        pe_at_day = r["pe"].reindex(pos.index, method="ffill")
        buys = pos[pos.diff() == 1].index
        sells = pos[pos.diff() == -1].index
        ax.scatter(buys, pe_at_day.reindex(buys), marker="^", color="green",
                   s=70, zorder=5, alpha=0.9, label="买入")
        ax.scatter(sells, pe_at_day.reindex(sells), marker="v", color="red",
                   s=70, zorder=5, alpha=0.9, label="卖出")
        ax.set_title(f"{r['name']} ({r['ticker']})", fontsize=12)
        ax.set_ylabel("P/E")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
