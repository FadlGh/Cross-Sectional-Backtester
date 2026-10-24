"""Run the main backtest and compare it with two baselines.

    python run.py                  # uses data/close.parquet, writes results/
    python run.py --cost-bps 20    # stress-test transaction costs

Compares:
  1. LightGBM long-short (the strategy)
  2. Single-factor momentum long-short (does the model beat a one-line rule?)
  3. Equal-weight market, long only (context only: different risk)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import build_dataset
from model import Config, walk_forward_predict
from portfolio import (information_coefficient, long_short_weights, metrics,
                       strategy_returns)


def format_table(table: pd.DataFrame) -> str:
    pct = {"ann_return", "ann_vol", "max_drawdown", "hit_rate"}
    cols = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "hit_rate", "weeks"]
    lines = ["| Strategy | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for name, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "weeks":
                cells.append(f"{int(v)}")
            elif c in pct:
                cells.append(f"{v:.1%}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/close.parquet")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    close = pd.read_parquet(args.data)
    df = build_dataset(close)

    cfg = Config()
    preds = walk_forward_predict(df, cfg)
    test = df.loc[preds.index]

    model = strategy_returns(long_short_weights(preds), test["fwd_ret"], args.cost_bps)
    momentum = strategy_returns(long_short_weights(test["mom_60"]), test["fwd_ret"],
                                args.cost_bps)
    market = test["fwd_ret"].groupby(level="date").mean()

    table = pd.DataFrame({
        "LightGBM long-short (net)": metrics(model["net"]),
        "LightGBM long-short (gross)": metrics(model["gross"]),
        "Momentum long-short (net)": metrics(momentum["net"]),
        "Equal-weight market (long only)": metrics(market),
    }).T

    ic = information_coefficient(preds, test["fwd_ret"])
    ic_t = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    dates = test.index.get_level_values("date")

    summary = "\n".join([
        "# Backtest summary",
        "",
        f"- Out-of-sample period: {dates.min().date()} to {dates.max().date()}",
        f"- Stocks: {test.index.get_level_values('ticker').nunique()}",
        f"- Transaction costs: {args.cost_bps:g} bps per unit traded",
        f"- Average weekly turnover: {model['turnover'].mean():.2f}",
        f"- Mean information coefficient: {ic.mean():.3f} (t-stat {ic_t:.2f})",
        "",
        format_table(table),
        "",
    ])
    (out / "summary.md").write_text(summary)

    weekly = pd.DataFrame({"model_net": model["net"], "model_gross": model["gross"],
                           "momentum_net": momentum["net"], "market": market})
    weekly.to_csv(out / "weekly_returns.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, label in [("model_net", "LightGBM long-short (net)"),
                       ("momentum_net", "Momentum long-short (net)"),
                       ("market", "Equal-weight market")]:
        ax.plot((1 + weekly[col]).cumprod(), label=label)
    ax.set_title("Out-of-sample growth of $1")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=150)

    print(summary)
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
