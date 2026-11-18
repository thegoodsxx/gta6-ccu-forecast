# GTA VI Twitch CCU Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thegoodsxx/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thegoodsxx/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/reward?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

## Method

Our entry for the **gta6-twitch-ccu-forecast-01** competition uses an advanced comparable launch curve matching and weighted empirical quantile model:

1. **Comparable Launch Curves**: For each comparable game launch, hourly data from `day_start` is extracted for hours 1 to 24 (`v(1)` to `v(24)`). Isolated missing hourly points are bridged via log-space interpolation. Comparables with missing hour-1 or excessive gaps are filtered out.
2. **Similarity Weighting**: Comparables are weighted relative to the target case's hour-1 viewer count (`p1`) using a log-space Gaussian similarity kernel (bandwidth $\sigma = 1.0$), combined with a platform boost for console-compatible launches (PlayStation / Xbox) and a recency preference.
3. **Quantile Bands**: For each release hour $h \in \{2 \dots 24\}$, the empirical distribution of log ratio curves $r(h) = v(h) / v(1)$ across the weighted comparable pool is computed. Weighted quantiles at levels $\tau \in \{0.05, 0.25, 0.50, 0.75, 0.95\}$ are scaled by anchor $p_1$ to construct robust quantile bands with a volatility floor.

## How to Run

Written in Python 3.13 (standard library only, deterministic, no network):

```bash
python3 main.py
```

It reads input cases from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes forecasted quantile JSON files to `/output` (or `$MAKEFUN_OUTPUT_DIR`).

## Validator & Leave-One-Out (LOO) Results

Validated using leave-one-out cross-validation across all comparables in the practice cases (`p00.in` to `p03.in`), scoring mean pinball loss on the log scale:

- **Overall Model LOO Loss**: `0.411350`
- **Overall House Baseline LOO Loss**: `0.929650`
- **Improvement over Baseline**: `55.75%`
