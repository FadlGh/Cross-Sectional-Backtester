# Cross-Sectional Equity Signal Research, with an Overfitting Audit

A weekly long-short equity strategy on 50 US large caps, built to answer two
questions honestly:

1. **Does a LightGBM model rank next week's relative winners better than a
   simple rule, after transaction costs?**
2. **If the best of many variants looks good, how much of that is luck?**

The second question is the point of the project. Trying many versions of a
strategy and reporting the best one is the most common way backtests lie.
This repo keeps every variant it tries and measures the damage with the
**Deflated Sharpe Ratio** and the **Probability of Backtest Overfitting**.

## Pipeline

```
data.py       download daily adjusted closes (yfinance), cache to data/
features.py   point-in-time features + forward-return labels, one row per (week, stock)
model.py      purged walk-forward LightGBM: train on the past, predict the next quarter, roll
portfolio.py  long top 20% / short bottom 20%, transaction costs, metrics
run.py        main backtest vs a momentum baseline and the market
audit.py      32 variants -> Deflated Sharpe Ratio + PBO (CSCV)
tests/        look-ahead (truncation) test, label timing, audit sanity checks
```

## Method

**Timeline of one decision.** Features are computed at the close of day *t*.
The position is entered at the close of *t+1* (you can't trade at the close
you used to decide) and exited at the close of *t+6*, one trading week later.
Features use only prices up to *t*; the label is the only thing that uses the
future.

**Features** (each a hypothesis about next week's relative return):

| Feature | Definition | Idea |
|---|---|---|
| `mom_5` | 5-day return | short-term reversal |
| `mom_20`, `mom_60` | 20- and 60-day return | momentum |
| `vol_20` | std of the last 20 daily returns | low-volatility effect |
| `ma_dist_50` | price / 50-day average - 1 | stretch vs trend |

**Cross-sectional ranking.** On each date every feature and the label are
converted to percentile ranks across stocks. The model learns *which stocks
beat the others*, not where the market goes; ranks stay comparable across
calm and volatile regimes; and a same-day rank cannot leak future
information (a global z-score would). Features are ranked *before* dropping
rows without a label, because whether a label exists depends on the future.

**Purged walk-forward.** Each model trains on the previous 3 years of weekly
samples and predicts the next 13 weeks, then rolls forward. A training row is
used only if its outcome was already known before the first test decision,
so the last week before each test block is purged.

**Portfolio.** Long the top 20% of predicted ranks, short the bottom 20%,
equal-weighted, +0.5 / -0.5 (gross exposure 1, market-neutral by
construction). Costs of 10 bps per unit of turnover are charged at each
weekly rebalance.

**Baselines.** A single-factor 60-day momentum long-short (does the model
beat a one-line rule?) and the equal-weight market (context only: it takes
market risk the long-short doesn't).

## Overfitting audit

`audit.py` runs a grid of 32 variants (4 feature sets x 2 tree sizes x
2 ensemble sizes x 2 training windows) on the same out-of-sample weeks and
keeps all of their returns.

- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014). Under the null
  that every variant has zero true Sharpe, the best of *N* trials is still
  expected to show a positive Sharpe. The DSR computes that expected maximum
  from the dispersion of the trials' Sharpe ratios and tests the best variant
  against it, adjusting for skewness and fat tails. Output: the probability
  that the best variant's true Sharpe is above zero.
- **Probability of Backtest Overfitting** (Bailey, Borwein, Lopez de Prado &
  Zhu, 2017), via combinatorially symmetric cross-validation. The weekly
  returns are cut into 16 blocks. For each of the C(16,8) = 12,870 ways to
  split them into two halves, the best variant on one half is located on the
  other half. PBO is the fraction of splits where the in-sample winner ranks
  at or below the out-of-sample median.

The tests check both on known cases: 30 pure-noise strategies give PBO near
50% and a low DSR; adding a genuine edge to one strategy gives PBO near 0
and DSR near 1.

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python data.py      # download prices
python -m pytest    # leakage and sanity tests
python run.py       # main backtest -> results/summary.md, equity_curve.png
python audit.py     # overfitting audit -> results/audit.md, variants.csv
```

## Results

See [`results/summary.md`](results/summary.md) for the main backtest and
[`results/audit.md`](results/audit.md) for the overfitting audit.

## Limitations

- **Survivorship bias.** The universe is today's large caps, chosen with
  hindsight. Stocks that fell out of the index are missing, which flatters
  long-only results and can distort the long-short.
- **Close-to-close execution.** Trades assume the closing price, no market
  impact, and no short-borrow fees.
- **Weight drift** within the week is ignored when computing turnover.
- **Correlated trials.** The 32 variants are far from independent, so the
  effective number of trials is lower than 32. This makes the DSR's luck
  benchmark conservative.
- **Small universe.** 50 stocks means each side of the portfolio holds 10,
  so returns are noisy.

## References

- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2017). The
  Probability of Backtest Overfitting. *Journal of Computational Finance*,
  20(4).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  (purged cross-validation)
