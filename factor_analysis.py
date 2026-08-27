import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = [
    "pe_level", "pe_pctile", "pe_z",
    "eps_yoy", "eps_accel", "eps_qoq",
    "mom_3m", "mom_6m", "mom_12m", "ma_dev",
]

FACTOR_NAME = {
    "pe_level": "P/E绝对值", "pe_pctile": "P/E百分位", "pe_z": "P/E的z-score",
    "eps_yoy": "EPS同比", "eps_accel": "EPS增速变化", "eps_qoq": "EPS环比",
    "mom_3m": "动量3月", "mom_6m": "动量6月", "mom_12m": "动量12月",
    "ma_dev": "均线偏离",
}


def load_db(path="factor_db.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["label"])
    return df


def cross_section_ic(df, factor):
    rows = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        ic = spearmanr(grp[factor], grp["label"])[0]
        rows.append({"date": date, "ic": ic})
    ic_series = pd.DataFrame(rows).set_index("date")["ic"]
    return ic_series


def ic_stats(ic_series):
    ic = ic_series.dropna()
    if len(ic) == 0:
        return dict(ic_mean=np.nan, icir=np.nan, t=np.nan, pos_ratio=np.nan)
    ic_mean = ic.mean()
    icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
    t = ic_mean / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan
    pos_ratio = (ic > 0).mean()
    return dict(ic_mean=ic_mean, icir=icir, t=t, pos_ratio=pos_ratio)


def quantile_analysis(df, factor, q=5):
    out = []
    for date, grp in df.groupby("date"):
        if len(grp) < q * 2:
            continue
        grp = grp.copy()
        grp["q"] = pd.qcut(grp[factor].rank(method="first"), q,
                            labels=False, duplicates="drop")
        for g, sub in grp.groupby("q"):
            out.append({"date": date, "q": g, "ret": sub["label"].mean()})
    qdf = pd.DataFrame(out)
    return qdf.groupby("q")["ret"].mean()


def main():
    print("加载因子库...")
    df = load_db()
    n_ticker = df["ticker"].nunique()
    n_rows = len(df)
    print(f"股票数: {n_ticker}, 样本数: {n_rows}, "
          f"日期 {df['date'].min().date()} ~ {df['date'].max().date()}\n")

    print(f"{'因子':<14}{'中文':<12}{'IC均值':>9}{'ICIR':>8}{'t值':>8}{'IC>0占比':>10}")
    print("=" * 62)
    results = {}
    for f in FEATURES:
        ic = cross_section_ic(df, f)
        s = ic_stats(ic)
        results[f] = s
        print(f"{f:<14}{FACTOR_NAME[f]:<12}{s['ic_mean']:>8.3f}{s['icir']:>8.2f}"
              f"{s['t']:>8.2f}{s['pos_ratio']:>9.1%}")

    print("\n因子单调性检验 (分5组, 各组平均下季收益):")
    print(f"{'因子':<14}" + "".join(f"{'Q'+str(i):>9}" for i in range(1, 6)) + f"{'多头-空头':>10}")
    for f in FEATURES:
        qr = quantile_analysis(df, f)
        if qr.empty:
            continue
        vals = [f"{v:>8.2%}" for v in qr.values]
        spread = qr.iloc[-1] - qr.iloc[0]
        print(f"{f:<14}" + "".join(vals) + f"{spread:>9.2%}")

    best = max(results.items(), key=lambda kv: abs(kv[1]["ic_mean"]))
    print(f"\n最强单因子: {best[0]} ({FACTOR_NAME[best[0]]}), IC均值 {best[1]['ic_mean']:.3f}")


if __name__ == "__main__":
    main()