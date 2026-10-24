"""Overfitting audit: how much of the best backtest is luck?

Runs a grid of strategy variants, keeps ALL of their returns, then measures:

  Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
      Probability that the best variant's true Sharpe is above zero, after
      correcting for the number of variants tried and for non-normal returns.

  Probability of Backtest Overfitting, PBO (Bailey, Borwein,
  Lopez de Prado & Zhu, 2017), via combinatorially symmetric
  cross-validation (CSCV)
      Split time into blocks; for every half/half split, pick the best
      variant on one half and check where it ranks on the other half.
      PBO = how often the in-sample winner is at or below the median
      out of sample.

    python audit.py    # writes results/audit.md and results/variants.csv
"""
import argparse
from itertools import combinations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from features import FEATURES, build_dataset
from model import Config, walk_forward_predict
from portfolio import PERIODS_PER_YEAR, long_short_weights, metrics, strategy_returns

EULER_GAMMA = 0.5772156649015329

FEATURE_SETS = {
    "all": FEATURES,
    "momentum_only": ["mom_5", "mom_20", "mom_60"],
    "no_vol": [f for f in FEATURES if f != "vol_20"],
    "no_reversal": [f for f in FEATURES if f != "mom_5"],
}


def make_variants() -> list[Config]:
    """4 feature sets x 2 tree sizes x 2 ensemble sizes x 2 train windows = 32."""
    grid = product(FEATURE_SETS.items(), [7, 31], [100, 300], [104, 156])
    first_test = 156  # same out-of-sample dates for every variant
    return [
        Config(name=f"{fs}_leaves{nl}_trees{nt}_train{tw}", features=list(feats),
               num_leaves=nl, n_estimators=nt, train_weeks=tw,
               first_test_week=first_test)
        for (fs, feats), nl, nt, tw in grid
    ]


def run_variants(df: pd.DataFrame, configs: list[Config],
                 cost_bps: float) -> pd.DataFrame:
    """Weekly net returns, one column per variant."""
    out = {}
    for i, cfg in enumerate(configs, 1):
        preds = walk_forward_predict(df, cfg)
        fwd = df.loc[preds.index, "fwd_ret"]
        out[cfg.name] = strategy_returns(long_short_weights(preds), fwd, cost_bps)["net"]
        print(f"[{i}/{len(configs)}] {cfg.name}")
    return pd.DataFrame(out)


def deflated_sharpe(returns: pd.DataFrame) -> dict:
    """Deflated Sharpe Ratio of the best column. Sharpe ratios are
    per-period (not annualized) inside the formula, as in the paper."""
    T, N = returns.shape
    srs = returns.mean() / returns.std(ddof=1)
    best = srs.idxmax()
    sr = srs[best]

    # Expected maximum Sharpe among N trials if every true Sharpe were zero.
    sr0 = np.sqrt(srs.var(ddof=1)) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / N)
        + EULER_GAMMA * norm.ppf(1 - 1 / (N * np.e))
    )

    r = returns[best].to_numpy()
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)  # 3 for a normal distribution
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)

    ann = np.sqrt(PERIODS_PER_YEAR)
    return {"best": best, "n_trials": N, "n_weeks": T,
            "best_sharpe_annual": sr * ann,
            "expected_max_sharpe_annual": sr0 * ann,
            "skew": g3, "kurtosis": g4, "dsr": norm.cdf(z)}


def _sharpe_from_sums(s1, s2, n):
    mean = s1 / n
    var = (s2 - n * mean**2) / (n - 1)
    return mean / np.sqrt(var)


def probability_of_backtest_overfitting(returns: pd.DataFrame,
                                        n_blocks: int = 16) -> dict:
    X = returns.to_numpy()
    T, N = X.shape
    X = X[: T - T % n_blocks]  # equal-sized blocks
    blocks = X.reshape(n_blocks, -1, N)
    block_len = blocks.shape[1]
    s1, s2 = blocks.sum(axis=1), (blocks**2).sum(axis=1)  # (blocks, N)
    tot1, tot2 = s1.sum(axis=0), s2.sum(axis=0)

    logits = []
    half = n_blocks // 2
    for combo in combinations(range(n_blocks), half):
        idx = list(combo)
        in1, in2 = s1[idx].sum(axis=0), s2[idx].sum(axis=0)
        sr_in = _sharpe_from_sums(in1, in2, half * block_len)
        sr_out = _sharpe_from_sums(tot1 - in1, tot2 - in2,
                                   (n_blocks - half) * block_len)
        winner = np.argmax(sr_in)
        rank = (sr_out < sr_out[winner]).sum() + 1  # 1 = worst, N = best
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))

    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_splits": len(logits),
            "logits": logits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/close.parquet")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = build_dataset(pd.read_parquet(args.data))
    returns = run_variants(df, make_variants(), args.cost_bps).dropna()

    table = pd.DataFrame({c: metrics(returns[c]) for c in returns}).T
    table.sort_values("sharpe", ascending=False).to_csv(out / "variants.csv")

    dsr = deflated_sharpe(returns)
    pbo = probability_of_backtest_overfitting(returns)

    report = "\n".join([
        "# Overfitting audit",
        "",
        f"- Variants tried: {dsr['n_trials']} (all kept, see variants.csv)",
        f"- Out-of-sample weeks: {dsr['n_weeks']}",
        f"- Best variant: `{dsr['best']}`",
        f"- Best annualized Sharpe (net): {dsr['best_sharpe_annual']:.2f}",
        f"- Expected best Sharpe from luck alone: {dsr['expected_max_sharpe_annual']:.2f}",
        f"- Deflated Sharpe Ratio: {dsr['dsr']:.3f} "
        "(probability the best variant's true Sharpe is above zero)",
        f"- Probability of backtest overfitting: {pbo['pbo']:.1%} "
        f"(over {pbo['n_splits']:,} CSCV splits)",
        "",
        "Reading it: DSR above 0.95 means the best result is unlikely to be "
        "luck. PBO near 50% means picking the in-sample winner is no better "
        "than picking at random.",
        "",
    ])
    (out / "audit.md").write_text(report)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pbo["logits"], bins=40)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(f"CSCV logits (PBO = {pbo['pbo']:.1%})")
    ax.set_xlabel("logit of out-of-sample rank of in-sample winner")
    fig.tight_layout()
    fig.savefig(out / "pbo_logits.png", dpi=150)

    print(report)


if __name__ == "__main__":
    main()
