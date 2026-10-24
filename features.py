"""Point-in-time features and forward-return labels.

Timeline for one sample at date t:
    close of t      features computed, decision made
    close of t+1    position entered (can't trade at the close you decided on)
    close of t+6    position exited, one trading week later

Features only use prices up to and including t. The label is the only
column allowed to use the future.
"""
import pandas as pd

FEATURES = ["mom_5", "mom_20", "mom_60", "vol_20", "ma_dist_50"]

ENTRY_LAG = 1        # enter at the close after the signal
HORIZON = 5          # hold for 5 trading days
REBALANCE_EVERY = 5  # one sample (and one rebalance) per week


def raw_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) feature tables. Positive shifts and trailing
    windows only, so every value at row t uses data up to t."""
    daily_ret = close.pct_change(fill_method=None)
    return {
        "mom_5": close / close.shift(5) - 1,    # short-term reversal
        "mom_20": close / close.shift(20) - 1,  # 1-month momentum
        "mom_60": close / close.shift(60) - 1,  # 3-month momentum
        "vol_20": daily_ret.rolling(20).std(),  # recent volatility
        "ma_dist_50": close / close.rolling(50).mean() - 1,  # stretch vs trend
    }


def forward_return(close: pd.DataFrame) -> pd.DataFrame:
    """Return from the close of t+1 to the close of t+6 (negative shifts =
    looking into the future, used ONLY for the label)."""
    entry = close.shift(-ENTRY_LAG)
    exit_ = close.shift(-(ENTRY_LAG + HORIZON))
    return exit_ / entry - 1


def exit_dates(index: pd.DatetimeIndex) -> pd.Series:
    """Date on which the outcome of a decision made at each date is known."""
    return pd.Series(index, index=index).shift(-(ENTRY_LAG + HORIZON))


def _to_long(wide: pd.DataFrame, dates: pd.Index) -> pd.Series:
    """Wide (date x ticker) -> long Series indexed by (date, ticker)."""
    s = wide.loc[dates].unstack()  # index: (ticker, date), NaNs kept
    s.index = s.index.set_names(["ticker", "date"])
    return s.swaplevel().sort_index()


def build_dataset(close: pd.DataFrame, drop_unlabeled: bool = True) -> pd.DataFrame:
    """One row per (rebalance date, ticker).

    Columns: ranked features, raw `fwd_ret` (for P&L), ranked `label`
    (what the model learns), and `label_end` (when the outcome is known,
    used to purge training data).
    """
    close = close.sort_index()
    dates = close.index[::REBALANCE_EVERY]

    cols = {name: _to_long(w, dates) for name, w in raw_features(close).items()}
    cols["fwd_ret"] = _to_long(forward_return(close), dates)
    df = pd.DataFrame(cols)

    # Drop the warm-up period and stocks with missing price history.
    df = df.dropna(subset=FEATURES)

    # Rank each feature across stocks on each date (0..1). Done BEFORE
    # dropping unlabeled rows: whether a stock has a label depends on the
    # future, so it must not decide which stocks get ranked.
    df[FEATURES] = df.groupby(level="date")[FEATURES].rank(pct=True)

    df["label_end"] = df.index.get_level_values("date").map(exit_dates(close.index))

    if drop_unlabeled:
        df = df.dropna(subset=["fwd_ret"])

    # Rank the label too: the model learns which stocks beat the others,
    # not where the whole market goes.
    df["label"] = df.groupby(level="date")["fwd_ret"].rank(pct=True)
    return df


if __name__ == "__main__":
    close = pd.read_parquet("data/close.parquet")
    df = build_dataset(close)
    n_dates = df.index.get_level_values("date").nunique()
    print(df.head())
    print(f"\n{len(df):,} rows, {n_dates} weekly dates, "
          f"{df.index.get_level_values('ticker').nunique()} tickers")
