"""Long-short portfolio construction, transaction costs and metrics."""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252 / 5  # weekly rebalancing


def long_short_weights(scores: pd.Series, quantile: float = 0.2) -> pd.Series:
    """Each date: long the top `quantile` of stocks, short the bottom.

    Equal weights within each side, +0.5 total long and -0.5 total short,
    so gross exposure is 1 and net market exposure is 0.
    """
    r = scores.groupby(level="date").rank(pct=True)
    long = (r > 1 - quantile).astype(float)
    short = (r <= quantile).astype(float)
    n_long = long.groupby(level="date").transform("sum")
    n_short = short.groupby(level="date").transform("sum")
    w = 0.5 * long / n_long - 0.5 * short / n_short
    return w.fillna(0.0).rename("weight")


def strategy_returns(weights: pd.Series, fwd_ret: pd.Series,
                     cost_bps: float = 10.0) -> pd.DataFrame:
    """Weekly gross return, turnover, cost and net return.

    Turnover = sum of |weight changes| between rebalances. Cost is charged
    at `cost_bps` per unit traded. Weight drift within the week is ignored.
    """
    gross = (weights * fwd_ret.reindex(weights.index)).groupby(level="date").sum()
    wide = weights.unstack("ticker").fillna(0.0)
    turnover = wide.diff().abs().sum(axis=1)
    turnover.iloc[0] = wide.iloc[0].abs().sum()
    cost = turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "turnover": turnover,
                         "cost": cost, "net": gross - cost})


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    ann_return = r.mean() * PERIODS_PER_YEAR
    ann_vol = r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    equity = (1 + r).cumprod()
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "hit_rate": (r > 0).mean(),
        "weeks": len(r),
    }


def information_coefficient(scores: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    """Per-date Spearman correlation between predicted and realized returns."""
    d = pd.DataFrame({"s": scores, "y": fwd_ret.reindex(scores.index)}).dropna()
    ranks = d.groupby(level="date").rank()
    return ranks.groupby(level="date").apply(lambda g: g["s"].corr(g["y"]))
