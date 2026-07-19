import sys
from pathlib import Path

import pandas as pd


from nullbt.research.regime_gate import gate_dates_from, make_gated_fn, market_entropy, rolling_direction_entropy

from nullbt.entropy import return_entropy


def test_rolling_entropy_matches_pointwise_return_entropy():
    """벡터화 rolling 구현이 검증된 순수 함수와 각 시점에서 일치해야 한다."""
    prices = [100.0]
    for i in range(60):
        prices.append(prices[-1] * (1.03 if (i * 7) % 3 else 0.98))
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    s = pd.Series(prices, index=idx)
    roll = rolling_direction_entropy(s, window=20)
    for t in [25, 40, len(prices) - 1]:
        expected = return_entropy(prices[: t + 1], window=20)
        assert abs(roll.iloc[t] - expected) < 1e-9


def test_gate_threshold_is_train_only():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    me = pd.Series([0.1] * 5 + [0.9] * 5, index=idx)
    train = list(idx[:5])  # train 전체가 0.1 → q70 임계 0.1
    gated, thr = gate_dates_from(me, train, 0.7)
    assert abs(thr - 0.1) < 1e-9
    assert gated == set(idx[5:])  # 0.9인 뒷구간만 게이트


def test_gated_fn_blocks_buy_only_on_gated_dates():
    def base(code, history, params):
        return ("BUY", 80.0)

    d_open = pd.DatetimeIndex(["2024-01-02"])
    d_gate = pd.DatetimeIndex(["2024-01-03"])
    fn = make_gated_fn(base, gated={d_gate[0]})
    assert fn("A", pd.DataFrame(index=d_open), {}) == ("BUY", 80.0)
    assert fn("A", pd.DataFrame(index=d_gate), {}) == ("HOLD", 0.0)


def test_market_entropy_is_cross_sectional_mean():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    up = pd.DataFrame({"close": [100.0 * (1.01**i) for i in range(30)]}, index=idx)
    price_data = {"A": up, "B": up.copy()}
    me = market_entropy(price_data, window=10)
    a = rolling_direction_entropy(up["close"], window=10)
    assert abs(me.iloc[-1] - a.iloc[-1]) < 1e-12  # 동일 종목 2개 평균 = 자기 자신
