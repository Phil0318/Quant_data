import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti"
]
matplotlib.rcParams["axes.unicode_minus"] = False


def moving_averages(price, short=20, long=60):
    short_ma = price.rolling(short).mean()
    long_ma = price.rolling(long).mean()
    return short_ma, long_ma


def backtest(price, short=20, long=60, fee=0.001):
    short_ma, long_ma = moving_averages(price, short, long)

    raw_position = (short_ma > long_ma).astype(float)
    position = raw_position.shift(1).fillna(0)

    ret = price.pct_change().fillna(0)
    strategy_ret = position * ret
    trade_cost = position.diff().abs().fillna(0) * fee
    strategy_ret = strategy_ret - trade_cost

    equity = (1 + strategy_ret).cumprod()
    return equity, position, short_ma, long_ma


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
    equity, position, short_ma, long_ma = backtest(price)

    print("== 动量交易策略回测 ==")
    print("参数: 双均线 short=20, long=60, 手续费=0.1%")
    for k, v in performance(equity).items():
        print(f"{k}: {v}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(price.index, price, label="收盘价", linewidth=1)
    axes[0].plot(short_ma.index, short_ma, label="短期均线(20)", linewidth=1, color="red")
    axes[0].plot(long_ma.index, long_ma, label="长期均线(60)", linewidth=1, color="blue")
    axes[0].set_title("价格与双均线")
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