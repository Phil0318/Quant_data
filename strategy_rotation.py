import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti"
]
matplotlib.rcParams["axes.unicode_minus"] = False


def ma_cross(price, short, long):
    short_ma = price.rolling(short).mean()
    long_ma = price.rolling(long).mean()
    return (short_ma > long_ma).astype(float)


def bollinger_mr(price, window=20, num_std=2.0):
    mid = price.rolling(window).mean()
    std = price.rolling(window).std()
    lower = mid - num_std * std
    pos = pd.Series(0, index=price.index)
    current = 0
    for i in range(len(price)):
        if np.isnan(mid.iloc[i]):
            continue
        if current == 0 and price.iloc[i] <= lower.iloc[i]:
            current = 1
        elif current == 1 and price.iloc[i] >= mid.iloc[i]:
            current = 0
        pos.iloc[i] = current
    return pos


def donchian(price, window=20):
    high = price.rolling(window).max().shift(1)
    low = price.rolling(window).min().shift(1)
    pos = pd.Series(0, index=price.index)
    current = 0
    for i in range(len(price)):
        if np.isnan(high.iloc[i]):
            continue
        if current == 0 and price.iloc[i] > high.iloc[i]:
            current = 1
        elif current == 1 and price.iloc[i] < low.iloc[i]:
            current = 0
        pos.iloc[i] = current
    return pos


def rsi_mr(price, window=14, oversold=30, overbought=70):
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    pos = pd.Series(0, index=price.index)
    current = 0
    for i in range(len(price)):
        if np.isnan(rsi.iloc[i]):
            continue
        if current == 0 and rsi.iloc[i] < oversold:
            current = 1
        elif current == 1 and rsi.iloc[i] > overbought:
            current = 0
        pos.iloc[i] = current
    return pos


def buy_hold(price):
    return pd.Series(1, index=price.index)


def cash(price):
    return pd.Series(0, index=price.index)


def ma_long_short(price, short=20, long=60):
    short_ma = price.rolling(short).mean()
    long_ma = price.rolling(long).mean()
    pos = pd.Series(0.0, index=price.index)
    valid = long_ma.notna()
    pos[valid & (short_ma > long_ma)] = 1
    pos[valid & (short_ma < long_ma)] = -1
    return pos


def trend_long_short(price, window=200):
    ma = price.rolling(window).mean()
    pos = (price > ma).astype(float) * 2 - 1
    return pos.where(ma.notna(), 0.0)


STRATEGIES = {
    "双均线20/60": lambda p: ma_cross(p, 20, 60),
    "双均线10/50": lambda p: ma_cross(p, 10, 50),
    "双均线50/200": lambda p: ma_cross(p, 50, 200),
    "布林带均值回归": lambda p: bollinger_mr(p, 20, 2.0),
    "唐奇安突破": lambda p: donchian(p, 20),
    "RSI均值回归": lambda p: rsi_mr(p, 14, 30, 70),
    "买入持有": buy_hold,
    "现金": cash,
    "双均线多空20/60": lambda p: ma_long_short(p, 20, 60),
    "趋势多空200": lambda p: trend_long_short(p, 200),
}


def trend_filter(price, window=200):
    return (price > price.rolling(window).mean()).astype(float)


def daily_return(price, position, fee=0.001):
    position = position.shift(1).fillna(0)
    ret = price.pct_change().fillna(0)
    cost = position.diff().abs().fillna(0) * fee
    return position * ret - cost


def _max_dd(x):
    cum = np.cumprod(1.0 + x)
    return float((cum / np.maximum.accumulate(cum) - 1.0).min())


def rolling_score(daily_ret, window, metric):
    if metric == "return":
        return (1 + daily_ret).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
    if metric == "sharpe":
        mu = daily_ret.rolling(window, min_periods=window).mean()
        sd = daily_ret.rolling(window, min_periods=window).std()
        return mu / sd * np.sqrt(252)
    if metric == "calmar":
        ret = (1 + daily_ret).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
        dd = daily_ret.rolling(window, min_periods=window).apply(_max_dd, raw=True)
        return ret / (-dd)
    raise ValueError(metric)


def select_best(strategy_rets, lookback=3, metric="sharpe"):
    window = lookback * 21
    scores = pd.DataFrame(index=strategy_rets.index)
    for name in strategy_rets.columns:
        scores[name] = rolling_score(strategy_rets[name], window, metric)
    month_end = scores.resample("ME").last().dropna(how="all")
    winners = month_end.idxmax(axis=1)
    winners.index = winners.index.to_period("M")
    active = winners.shift(1)
    daily_period = strategy_rets.index.to_period("M")
    active_daily = pd.Series(active.reindex(daily_period).values, index=strategy_rets.index)
    active_daily = active_daily.fillna("买入持有")
    port_ret = pd.Series(index=strategy_rets.index, dtype=float)
    for name in strategy_rets.columns:
        mask = active_daily == name
        port_ret.loc[mask] = strategy_rets.loc[mask, name]
    return port_ret, active_daily


def performance(equity):
    total_ret = equity.iloc[-1] - 1
    ann_ret = equity.iloc[-1] ** (252 / len(equity)) - 1
    daily_ret = equity.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252 - 0.02) / vol
    peak = equity.cummax()
    max_dd = (equity / peak - 1).min()
    return {
        "总收益": total_ret,
        "年化收益": ann_ret,
        "年化波动": vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
    }


def run(price, lookback, metric, use_trend, trend_window=200, fee=0.001):
    positions = {name: func(price) for name, func in STRATEGIES.items()}
    if use_trend:
        trend = trend_filter(price, trend_window)
        for name in positions:
            pos = positions[name]
            positions[name] = pos.where(pos <= 0, pos * trend)
    strategy_rets = pd.DataFrame(
        {name: daily_return(price, pos, fee) for name, pos in positions.items()},
        index=price.index,
    )
    port_ret, active_daily = select_best(strategy_rets, lookback, metric)
    port_equity = (1 + port_ret).cumprod()
    return port_equity, active_daily, strategy_rets


def download_price(ticker="QQQ", period="10y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


def main():
    ticker = input("输入代码 (默认 QQQ): ").strip() or "QQQ"
    price = download_price(ticker)
    if price.empty:
        print(f"未获取到 {ticker} 的数据，请检查代码是否正确")
        return

    bh = performance((1 + price.pct_change().fillna(0)).cumprod())

    print(f"== 策略轮换回测 ({ticker}) ==")
    print(f"基准(买入持有): 总收益 {bh['总收益']:.1%}  夏普 {bh['夏普比率']:.2f}  "
          f"最大回撤 {bh['最大回撤']:.1%}\n")

    results = []
    for metric in ["return", "sharpe", "calmar"]:
        for lookback in [1, 3, 6, 12]:
            for use_trend in [False, True]:
                eq, _, _ = run(price, lookback, metric, use_trend)
                p = performance(eq)
                results.append({
                    "回看(月)": lookback,
                    "选优指标": metric,
                    "趋势过滤": "开" if use_trend else "关",
                    "总收益": p["总收益"],
                    "年化": p["年化收益"],
                    "夏普": p["夏普比率"],
                    "最大回撤": p["最大回撤"],
                    "equity": eq,
                })

    df = pd.DataFrame(results)
    best = max(results, key=lambda r: r["夏普"])

    print(f"{'回看(月)':<8}{'选优指标':<10}{'趋势过滤':<8}{'总收益':>10}{'年化':>9}"
          f"{'夏普':>8}{'最大回撤':>10}")
    for r in results:
        mark = " *" if r is best else ""
        print(f"{r['回看(月)']:<8}{r['选优指标']:<10}{r['趋势过滤']:<8}"
              f"{r['总收益']:>9.1%}{r['年化']:>8.1%}{r['夏普']:>8.2f}"
              f"{r['最大回撤']:>9.1%}{mark}")

    print(f"\n最优配置: 回看 {best['回看(月)']} 月 / {best['选优指标']} / "
          f"趋势过滤{best['趋势过滤']}")

    eq, active_daily, strategy_rets = run(
        price, best["回看(月)"], best["选优指标"],
        best["趋势过滤"] == "开",
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for name in STRATEGIES:
        s_eq = (1 + strategy_rets[name]).cumprod()
        axes[0].plot(s_eq.index, s_eq, linewidth=0.8, alpha=0.45, label=name)
    axes[0].plot(eq.index, eq, linewidth=2.2, color="black", label="轮换组合")
    axes[0].plot(price.index, price / price.iloc[0], linewidth=1.2, color="gray",
                 linestyle="--", label="买入持有")
    axes[0].set_title("各策略净值与最优轮换组合")
    axes[0].legend(loc="upper left", fontsize=8)

    cat_map = {name: i for i, name in enumerate(STRATEGIES)}
    active_cat = active_daily.map(cat_map)
    axes[1].step(active_cat.index, active_cat, where="post", linewidth=1.5, color="purple")
    axes[1].set_yticks(list(cat_map.values()))
    axes[1].set_yticklabels(list(cat_map.keys()), fontsize=8)
    axes[1].set_ylim(-0.5, len(STRATEGIES) - 0.5)
    axes[1].set_title("每月执行中的策略")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()