import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf

from pe_strategy import get_pe_history, pe_linear_strategy, performance

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

LOW_PCT = 50
HIGH_PCT = 75
PERIOD = "6y"


def get_pctile(price, pe, lookback_years=5):
    window = lookback_years * 4
    pctile = pe["pe"].rolling(window, min_periods=window).rank(pct=True) * 100
    return pctile.reindex(price.index, method="ffill").ffill().bfill().fillna(50)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, (ticker, name) in enumerate(TICKERS):
        try:
            pe = get_pe_history(ticker, name)
        except Exception as e:
            print(f"{ticker} PE fail: {e}")
            time.sleep(3)
            continue

        px = yf.Ticker(ticker).history(period=PERIOD, auto_adjust=True)["Close"]
        px.index = px.index.tz_localize(None)
        px = px[px.index >= pe.index[0]]

        eq, bh_eq, position, pctile = pe_linear_strategy(
            px, pe, low_pct=LOW_PCT, high_pct=HIGH_PCT)

        s = performance(eq)
        b = performance(bh_eq)

        ax = axes[i]
        pe_at_day = pe["pe"].reindex(px.index, method="ffill")
        ax.plot(pe_at_day.index, pe_at_day, linewidth=1.3, color="steelblue",
                label="P/E")

        pct = get_pctile(px, pe)
        enter_full = pct.index[(pct.shift(1) > LOW_PCT) & (pct <= LOW_PCT)]
        enter_flat = pct.index[(pct.shift(1) < HIGH_PCT) & (pct >= HIGH_PCT)]
        ax.scatter(enter_full, pe_at_day.reindex(enter_full), marker="^",
                   color="green", s=80, zorder=5, label=f"满仓(≤{LOW_PCT}%)")
        ax.scatter(enter_flat, pe_at_day.reindex(enter_flat), marker="v",
                   color="red", s=80, zorder=5, label=f"空仓(≥{HIGH_PCT}%)")

        ax2 = ax.twinx()
        ax2.plot(eq.index, eq, linewidth=1.5, color="orange", label="策略净值")
        ax2.plot(bh_eq.index, bh_eq, linewidth=1.0, color="gray",
                 linestyle="--", alpha=0.7, label="买持净值")
        ax2.set_yscale("log")

        ax.set_ylabel("P/E")
        ax2.set_ylabel("净值 (log)")
        ax.set_title(f"{TICKER_NAMES[ticker]} ({ticker})\n"
                     f"策略年化 {s['年化收益']:.1%} / 夏普 {s['夏普比率']:.2f}   |   "
                     f"买持年化 {b['年化收益']:.1%} / 夏普 {b['夏普比率']:.2f}",
                     fontsize=10)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

        time.sleep(2)

    fig.suptitle(f"P/E 线性仓位策略 ({LOW_PCT}%~{HIGH_PCT}%减仓区间, 近6年)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("/tmp/pe_strategy_result.png", dpi=110)
    print("图已保存到 /tmp/pe_strategy_result.png")
    plt.show()


if __name__ == "__main__":
    main()