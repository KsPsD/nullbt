import numpy as np
import pandas as pd
import pytest

from nullbt import BootstrapResult, bootstrap_metrics
from nullbt.bootstrap import _stationary_indices


def _returns(n=250, mu=0.0008, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def test_returns_bootstrap_result_type():
    res = bootstrap_metrics(_returns(), n_resamples=100, seed=1)
    assert isinstance(res, BootstrapResult)
    assert res.n_resamples == 100
    assert set(res.samples) == {"sharpe", "cagr", "max_drawdown"}
    assert len(res.samples["sharpe"]) == 100


def test_deterministic_same_seed():
    a = bootstrap_metrics(_returns(), n_resamples=100, seed=42)
    b = bootstrap_metrics(_returns(), n_resamples=100, seed=42)
    assert a.sharpe.low == b.sharpe.low
    assert a.sharpe.high == b.sharpe.high
    assert np.array_equal(a.samples["sharpe"], b.samples["sharpe"])


def test_different_seed_differs():
    a = bootstrap_metrics(_returns(), n_resamples=100, seed=1)
    b = bootstrap_metrics(_returns(), n_resamples=100, seed=2)
    assert not np.array_equal(a.samples["sharpe"], b.samples["sharpe"])


def test_ci_ordering():
    res = bootstrap_metrics(_returns(), n_resamples=300, seed=3)
    for m in (res.sharpe, res.cagr, res.max_drawdown):
        assert m.low <= m.high
        assert m.low <= m.mean <= m.high


def test_constant_returns_collapse_sharpe():
    # 표준편차 0 → Sharpe는 모든 재표본에서 0, CI도 0으로 붕괴.
    res = bootstrap_metrics([0.001] * 100, n_resamples=50, expected_block=10, seed=0)
    assert res.sharpe.low == 0.0
    assert res.sharpe.high == 0.0


def test_nan_dropped():
    res = bootstrap_metrics([0.01, np.nan, -0.02, 0.03, np.nan, 0.01],
                            n_resamples=20, expected_block=2, seed=0)
    assert isinstance(res, BootstrapResult)


def test_bootstrap_mean_near_point_for_iid():
    # iid 수익률이면 부트스트랩 Sharpe 평균이 점추정 근처여야(통계적 정합성).
    res = bootstrap_metrics(_returns(n=500, seed=7), n_resamples=800,
                            expected_block=5, seed=7)
    assert abs(res.sharpe.mean - res.sharpe.point) < 0.5


@pytest.mark.parametrize("bad", [
    dict(daily_returns=[0.01]),                       # 너무 짧음
    dict(daily_returns=[0.01, 0.02], expected_block=5),  # 블록 > n
    dict(daily_returns=[0.01, 0.02], ci_level=1.5),   # ci 범위 밖
    dict(daily_returns=[0.01, 0.02], n_resamples=0),  # 재표본 0
])
def test_input_validation(bad):
    with pytest.raises(ValueError):
        bootstrap_metrics(**bad)


def test_stationary_indices_in_range_and_length():
    rng = np.random.default_rng(0)
    idx = _stationary_indices(50, 10, rng)
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def test_format_report_runs():
    res = bootstrap_metrics(_returns(), n_resamples=50, seed=0)
    txt = res.format_report()
    assert "Sharpe" in txt and "CI" in txt
