import pandas as pd

from nullbt.engine import BacktestConfig
from nullbt.explore.search import evaluate_params, run_search
from nullbt.explore.spec import StrategySpec


def _df(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {"open": prices, "high": [p * 1.02 for p in prices], "low": [p * 0.98 for p in prices],
         "close": prices, "volume": [1000] * len(prices)},
        index=idx,
    )


def _buy_if_cutoff(code, history, params):
    # confidence_cutoff 이상이면 BUY (탐색이 cutoff를 조절하도록)
    return ("BUY", 30.0)


_PRICE = {"A": _df([100 + i for i in range(40)])}
_WINDOWS = [list(_PRICE["A"].index)]
_SPEC = StrategySpec(
    constraints={"max_mdd": 0.5, "min_trades": 1},
    search_space={"confidence_cutoff": (20.0, 50.0), "stop_pct": (0.03, 0.10),
                  "target_pct": (0.05, 0.25), "max_hold_days": (3, 10)},
)


def test_evaluate_params_returns_score_and_metrics():
    params = {"confidence_cutoff": 20.0, "stop_pct": 0.05, "target_pct": 0.1, "max_hold_days": 5}
    score, metrics = evaluate_params(
        params, _WINDOWS, _PRICE, _buy_if_cutoff, _SPEC, BacktestConfig(max_positions=1, capital=1_000_000)
    )
    assert isinstance(score, float)
    assert "sharpe" in metrics and "mdd" in metrics and "trades" in metrics


def test_run_search_returns_study_with_best_params():
    study = run_search(
        _SPEC, _PRICE, _WINDOWS, _buy_if_cutoff,
        BacktestConfig(max_positions=1, capital=1_000_000), n_trials=5,
    )
    assert study.best_params is not None
    assert set(study.best_params) >= {"confidence_cutoff", "stop_pct", "target_pct", "max_hold_days"}
