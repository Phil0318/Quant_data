import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti"
]
matplotlib.rcParams["axes.unicode_minus"] = False


def bollinger_bands(price, window=20, num_std=2.0):
    mid = price.rolling(window).mean()
    std = price.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def backtest(price, window=20, num_std=2.0, fee=0.001):
    mid, upper, lower = bollinger_bands(price, window, num_std)

    position = pd.Series(0, index=price.index)
    entry_price = pd.Series(np.nan, index=price.index)
    current = 0
    last_entry = np.nan

    for i in range(len(price)):
        if np.isnan(mid.iloc[i]):
            continue
        if current == 0 and price.iloc[i] <= lower.iloc[i]:
            current = 1
            last_entry = price.iloc[i]
        elif current == 1 and price.iloc[i] >= mid.iloc[i]:
            current = 0
        position.iloc[i] = current
        entry_price.iloc[i] = last_entry

    position = position.shift(1).fillna(0)
    ret = price.pct_change().fillna(0)
    strategy_ret = position * ret
    trade_cost = position.diff().abs().fillna(0) * fee
    strategy_ret = strategy_ret - trade_cost

    equity = (1 + strategy_ret).cumprod()
    return equity, position, entry_price, mid, upper, lower


def performance(equity):
    total_ret = equity.iloc[-1] - 1
    ann_ret = equity.iloc[-1] ** (252 / len(equity)) - 1
    daily_ret = equity.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252 - 0.02) / vol
    peak = equity.cummax()
    max_dd = (equity / peak - 1).min()
    return {
        "总收益": f"{total_ret:.2%}",
        "年化收益": f"{ann_ret:.2%}",
        "年化波动": f"{vol:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd:.2%}",
    }


def download_price(ticker="600519.SS", period="5y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


def main():
    ticker = input("输入代码 (默认 600519.SS 贵州茅台): ").strip() or "600519.SS"
    price = download_price(ticker)
    if price.empty:
        print(f"未获取到 {ticker} 的数据，请检查代码是否正确 (A股需加 .SS/.SZ 后缀)")
        return
    equity, position, entry, mid, upper, lower = backtest(price)

    print("== 均值回归策略回测 ==")
    print("参数: 布林带 window=20, num_std=2.0, 手续费=0.1%")
    for k, v in performance(equity).items():
        print(f"{k}: {v}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(price.index, price, label="收盘价", linewidth=1)
    axes[0].plot(mid.index, mid, label="中轨", linewidth=1, color="gray", linestyle="--")
    axes[0].plot(upper.index, upper, label="上轨", linewidth=1, color="orange", linestyle="--")
    axes[0].plot(lower.index, lower, label="下轨", linewidth=1, color="purple", linestyle="--")
    axes[0].fill_between(price.index, lower, upper, alpha=0.05, color="gray")
    axes[0].set_title("价格与布林带")
    axes[0].legend(loc="upper left")

    buy = price[position.diff() == 1]
    sell = price[position.diff() == -1]
    axes[0].scatter(buy.index, buy, marker="^", color="red", s=60, label="买入")
    axes[0].scatter(sell.index, sell, marker="v", color="green", s=60, label="卖出")

    axes[1].plot(position.index, position, label="仓位", linewidth=1)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title("仓位")
    axes[1].legend(loc="upper left")

    axes[2].plot(equity.index, equity, label="策略净值", linewidth=1.5)
    axes[2].plot(price.index, price / price.iloc[0], label="买入持有", linewidth=1, alpha=0.6)
    axes[2].set_title("净值曲线")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()