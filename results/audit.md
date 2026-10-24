# Overfitting audit

- Variants tried: 32 (all kept, see variants.csv)
- Out-of-sample weeks: 476
- Best variant: `no_reversal_leaves7_trees100_train156`
- Best annualized Sharpe (net): -0.94
- Expected best Sharpe from luck alone: 0.55
- Deflated Sharpe Ratio: 0.000 (probability the best variant's true Sharpe is above zero)
- Probability of backtest overfitting: 20.9% (over 12,870 CSCV splits)
- Variants with positive gross Sharpe (before costs): 3 of 32
- Rank correlation across variants, turnover vs net Sharpe: -0.60
- Rank correlation across variants, gross vs net Sharpe: +0.86

Reading it: DSR above 0.95 means the best result is unlikely to be luck. PBO near 50% means picking the in-sample winner is no better than picking at random.
