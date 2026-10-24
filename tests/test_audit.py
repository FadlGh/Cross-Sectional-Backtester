import numpy as np
import pandas as pd

from audit import deflated_sharpe, probability_of_backtest_overfitting


def noise_strategies(n_weeks=480, n_strategies=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, size=(n_weeks, n_strategies)),
                        columns=[f"s{i}" for i in range(n_strategies)])


def test_pure_noise_is_flagged_as_overfit():
    r = noise_strategies()
    assert 0.3 < probability_of_backtest_overfitting(r)["pbo"] < 0.7
    assert deflated_sharpe(r)["dsr"] < 0.95


def test_real_edge_is_not_flagged():
    r = noise_strategies()
    r["s0"] += 0.004  # one strategy with a genuine positive edge
    assert probability_of_backtest_overfitting(r)["pbo"] < 0.1
    assert deflated_sharpe(r)["dsr"] > 0.95
