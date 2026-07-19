import numpy as np
import pandas as pd

from nullbt.metrics import (
    cagr,
    deflated_sharpe,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)


def test_sharpe_zero_for_constant_returns():
    r = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert sharpe_ratio(r) == 0.0


def test_sharpe_positive_for_upward_returns():
    r = pd.Series([0.01, 0.02, 0.01, 0.015])
    assert sharpe_ratio(r) > 0


def test_max_drawdown_known_sequence():
    equity = pd.Series([100, 120, 90, 110])  # peak 120 -> trough 90 = -25%
    assert abs(max_drawdown(equity) - 0.25) < 1e-9


def test_max_drawdown_monotonic_up_is_zero():
    equity = pd.Series([100, 110, 120])
    assert max_drawdown(equity) == 0.0


def test_win_rate():
    assert win_rate([0.1, -0.05, 0.2, -0.01]) == 0.5
    assert win_rate([]) == 0.0


def test_deflated_sharpe_decreases_with_more_trials():
    base = deflated_sharpe(2.0, n_trials=2, n_obs=252)
    many = deflated_sharpe(2.0, n_trials=500, n_obs=252)
    assert 0.0 <= many <= base <= 1.0


def test_deflated_sharpe_decreases_with_higher_sharpe_std():
    # 교차-trial Sharpe 분산이 클수록 SR0 허들↑ → DSR↓ (deflation 강화).
    # 하드코딩 1.0 대신 실제 분산을 넘기는 것이 의미 있음을 보장.
    low_std = deflated_sharpe(2.0, n_trials=50, n_obs=252, sharpe_std=0.5)
    high_std = deflated_sharpe(2.0, n_trials=50, n_obs=252, sharpe_std=2.0)
    assert high_std < low_std
