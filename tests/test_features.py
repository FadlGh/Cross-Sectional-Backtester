import numpy as np
import pandas as pd

from features import ENTRY_LAG, FEATURES, HORIZON, build_dataset


def synthetic_close(n_days=800, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.015, size=(n_days, n_tickers))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates,
                         columns=[f"T{i}" for i in range(n_tickers)])
    close.iloc[300:303, 2] = np.nan  # a short gap in one stock
    return close


def test_no_lookahead_in_features():
    """Truncation test: deleting data after date D must not change any
    feature, or which rows exist, at dates up to D."""
    close = synthetic_close()
    for cut_at in [497, 500, 503]:  # cut on and between rebalance dates
        cut_date = close.index[cut_at]
        full = build_dataset(close, drop_unlabeled=False)
        cut = build_dataset(close.loc[:cut_date], drop_unlabeled=False)

        full_part = full[full.index.get_level_values("date") <= cut_date]
        assert len(cut) > 0
        pd.testing.assert_index_equal(full_part.index, cut.index)
        pd.testing.assert_frame_equal(full_part[FEATURES], cut[FEATURES])


def test_label_timing():
    """fwd_ret at t is the return from close t+1 to close t+6."""
    close = synthetic_close()
    df = build_dataset(close)
    date, ticker = df.index[100]
    i = close.index.get_loc(date)
    expected = (close[ticker].iloc[i + ENTRY_LAG + HORIZON]
                / close[ticker].iloc[i + ENTRY_LAG] - 1)
    assert np.isclose(df.loc[(date, ticker), "fwd_ret"], expected)
    assert df.loc[(date, ticker), "label_end"] == close.index[i + ENTRY_LAG + HORIZON]


def test_ranks_are_in_unit_interval():
    df = build_dataset(synthetic_close())
    cols = FEATURES + ["label"]
    assert ((df[cols] > 0) & (df[cols] <= 1)).all().all()
