import pandas as pd

from nullbt.explore.cli import run_exploration


def _df(prices, volumes=None):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    if volumes is None:
        volumes = [1000] * len(prices)
    return pd.DataFrame(
        {"open": prices, "high": [p * 1.02 for p in prices], "low": [p * 0.98 for p in prices],
         "close": prices, "volume": volumes},
        index=idx,
    )


def test_run_exploration_end_to_end_returns_report():
    price = {"A": _df([100 + i for i in range(80)]), "B": _df([200 - i * 0.5 for i in range(80)])}
    out = run_exploration(price, n_trials=4, n_folds=2, top_k=3)
    assert isinstance(out, str)
    assert "OOS" in out and "Holdout" in out


def test_run_exploration_default_spec_trades_nonzero():
    """confidence_threshold in search_space → real adapter must produce trades for all params.

    Data is designed so that day 50 always fires a BUY for every param combination:
      - Days 0-49: flat price 100, normal volume 1000  (warms up MA20)
      - Day 50: price jumps to 115 (>MA20*1.10≈110.8) with volume 5000, high set == close
        → envelope breakout (+25) + 52week high (+15, because close==high==max_high so
          close >= 0.99*max(high) holds) + volume surge 4.6x (+15) = score 55
        score 55 >= any confidence_threshold in [20,55] and conf 55 >= any cutoff in [20,50]
        → BUY fires deterministically regardless of Optuna's parameter choice
      - Days 51-149: continue rising to ensure trade can close via target or max_hold
    """
    # Days 0-49: flat at 100
    flat_prices = [100] * 50
    flat_vols = [1000] * 50
    # Day 50: breakout jump with volume spike
    breakout_prices = [115]
    breakout_vols = [5000]
    # Days 51-149: gradual rise
    rise_prices = [115 + i for i in range(1, 100)]
    rise_vols = [1000] * 99

    prices = flat_prices + breakout_prices + rise_prices
    volumes = flat_vols + breakout_vols + rise_vols
    df = _df(prices, volumes)
    # Make the breakout day a true 52-week high: high must equal close so that
    # close >= 0.99 * max(high) is satisfied (the default high=close*1.02 defeats this check).
    df.loc[df.index[50], "high"] = df.loc[df.index[50], "close"]
    price_data = {"A": df}

    out = run_exploration(price_data, n_trials=5, n_folds=2, top_k=3)

    # Extract trade counts from Trd column (appears right after MDD% in each data row)
    lines = out.split("\n")[2:]  # skip header + separator
    trade_counts = []
    for line in lines:
        if "%" in line:
            right = line.split("%", 1)[1]
            tokens = right.split()
            if tokens:
                try:
                    trade_counts.append(int(tokens[0]))
                except ValueError:
                    pass

    assert any(t > 0 for t in trade_counts), (
        f"Expected at least one candidate with trades>0 but got {trade_counts}.\n"
        f"Full report:\n{out}"
    )
