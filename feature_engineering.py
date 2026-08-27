import time
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

from pe_strategy import get_pe_history

TICKERS = [
    ("AAPL", "apple"), ("MSFT", "microsoft"), ("GOOG", "alphabet"),
    ("AMZN", "amazon"), ("NVDA", "nvidia"), ("META", "meta-platforms"),
]

FEATURE_COLS = [
    "pe_level", "pe_pctile", "pe_z",
    "eps_yoy", "eps_accel", "eps_qoq",
    "mom_3m", "mom_6m", "mom_12m", "ma_dev",
]


def build_features(ticker, name):
    pe = get_pe_history(ticker, name).sort_index()
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

    df = pd.DataFrame({
        "pe_level": pe_level, "pe_pctile": pe_pctile, "pe_z": pe_z,
        "eps_yoy": eps_yoy, "eps_accel": eps_accel, "eps_qoq": eps_qoq,
        "mom_3m": mom_3m, "mom_6m": mom_6m, "mom_12m": mom_12m, "ma_dev": ma_dev,
        "label": label,
    })
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    ratio_cols = ["pe_z", "eps_yoy", "eps_accel", "eps_qoq",
                  "mom_3m", "mom_6m", "mom_12m", "ma_dev"]
    df[ratio_cols] = df[ratio_cols].clip(-5, 5)
    df["label"] = df["label"].clip(-5, 5)
    df["ticker"] = ticker
    return df


def main():
    print("构建特征...")
    frames = []
    for ticker, name in TICKERS:
        try:
            df = build_features(ticker, name)
            frames.append(df)
            print(f"  {ticker}: {len(df)} 个季度样本")
        except Exception as e:
            print(f"  {ticker}: 失败 {e}")
        time.sleep(2)

    data = pd.concat(frames).sort_index()
    print(f"\n总样本: {len(data)}, 时间范围 {data.index[0].date()} ~ {data.index[-1].date()}")

    cut = data.index[int(len(data) * 0.7)]
    train = data[data.index < cut]
    test = data[data.index >= cut]
    print(f"训练集: {len(train)} (到 {cut.date()}), 测试集: {len(test)}")

    X_tr, y_tr = train[FEATURE_COLS], train["label"]
    X_te, y_te = test[FEATURE_COLS], test["label"]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print("\n训练 MLP (3层: 64-32-16)...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16), activation="relu",
        max_iter=1000, random_state=42, early_stopping=False,
    )
    model.fit(X_tr_s, y_tr)

    pred_tr = model.predict(X_tr_s)
    pred_te = model.predict(X_te_s)

    ic_tr = spearmanr(pred_tr, y_tr)[0]
    ic_te = spearmanr(pred_te, y_te)[0]

    print(f"\n{'='*60}")
    print(f"信息系数 IC (预测 vs 实际下一季收益的秩相关)")
    print(f"  训练集 IC: {ic_tr:.3f}")
    print(f"  测试集 IC: {ic_te:.3f}")
    print(f"{'='*60}")

    if abs(ic_te) < 0.05:
        print("结论: 测试集 IC 接近 0，MLP 未能学到可泛化的预测能力（典型的小样本金融问题）")
    elif ic_te > 0.05:
        print("结论: 测试集 IC 为正，有一定预测能力")
    else:
        print("结论: 测试集 IC 为负，模型学到的反而是反指（可能过拟合或市场反转）")

    feat_imp = pd.Series(
        np.abs(model.coefs_[0]).mean(axis=1), index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(f"\n特征重要性 (第一层权重绝对值的均值):")
    for f, v in feat_imp.items():
        print(f"  {f:<12} {v:.4f}")


if __name__ == "__main__":
    main()