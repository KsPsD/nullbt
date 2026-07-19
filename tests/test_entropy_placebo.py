import sys
from pathlib import Path

import pandas as pd


from nullbt.research.entropy_placebo import make_placebo_fn


def _base_fn(code, history, params):
    return ("BUY", 80.0)


def _hist(day: str) -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([day]))


def test_placebo_deterministic_per_seed():
    fn = make_placebo_fn(_base_fn, 0.5, seed=7)
    a = fn("005930", _hist("2024-01-02"), {})
    b = fn("005930", _hist("2024-01-02"), {})
    assert a == b  # 같은 (seed, code, date) → 같은 결정


def test_placebo_suppression_rate_approx():
    fn = make_placebo_fn(_base_fn, 0.86, seed=0)
    days = pd.date_range("2020-01-01", periods=500, freq="D")
    kept = sum(1 for d in days if fn("000100", _hist(str(d.date())), {})[0] == "BUY")
    assert 0.05 < kept / 500 < 0.25  # 억제율 ~0.86 → 유지율 ~0.14


def test_placebo_passes_through_hold():
    fn = make_placebo_fn(lambda c, h, p: ("HOLD", 0.0), 1.0, seed=0)
    assert fn("005930", _hist("2024-01-02"), {}) == ("HOLD", 0.0)
