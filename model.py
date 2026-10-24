"""Purged walk-forward training with LightGBM.

Every prediction is made by a model trained only on the past:
    train on the previous `train_weeks` weeks -> predict the next
    `test_weeks` weeks -> roll forward and repeat.

Purging: a training row is only usable if its outcome (label_end) was
already known before the first decision of the test block. Without this,
the last training rows' labels would overlap the test period, which
would be impossible in live trading.
"""
from dataclasses import dataclass, field

import pandas as pd
from lightgbm import LGBMRegressor

from features import FEATURES


@dataclass
class Config:
    name: str = "base"
    features: list = field(default_factory=lambda: list(FEATURES))
    train_weeks: int = 156       # ~3 years of history per model
    test_weeks: int = 13         # retrain every quarter
    first_test_week: int | None = None  # defaults to train_weeks
    purge: bool = True
    num_leaves: int = 15
    n_estimators: int = 200
    learning_rate: float = 0.05
    min_child_samples: int = 100
    seed: int = 0

    def make_model(self) -> LGBMRegressor:
        return LGBMRegressor(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
        )


def walk_forward_predict(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Out-of-sample predictions indexed by (date, ticker)."""
    dates = df.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    first = cfg.train_weeks if cfg.first_test_week is None else cfg.first_test_week

    preds = []
    for start in range(first, len(unique_dates), cfg.test_weeks):
        test_dates = unique_dates[start:start + cfg.test_weeks]
        train_dates = unique_dates[max(0, start - cfg.train_weeks):start]

        train = df[dates.isin(train_dates)]
        if cfg.purge:
            train = train[train["label_end"] < test_dates[0]]
        test = df[dates.isin(test_dates)]

        model = cfg.make_model()
        model.fit(train[cfg.features], train["label"])
        preds.append(pd.Series(model.predict(test[cfg.features]), index=test.index))

    return pd.concat(preds).rename("pred")
