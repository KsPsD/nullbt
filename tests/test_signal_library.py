import pandas as pd

from nullbt.signals import (
    entropy_damped,
    make_dipbuy_signal_fn,
    make_meanrev_signal_fn,
)


def _df(closes, volumes=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": volumes if volumes is not None else [1000] * len(closes),
        },
        index=idx,
    )


def test_meanrev_buys_deep_dip_in_uptrend():
    # 장기 상승추세 + 마지막에 급락 → z 깊은 음수 → BUY
    closes = [100.0 + i * 0.5 for i in range(120)]
    closes[-1] = closes[-10] - 8.0  # 급락(추세선 위는 유지)
    fn = make_meanrev_signal_fn()
    action, conf = fn("A", _df(closes), {"mr_window": 10, "z_entry": 1.0, "trend_window": 100})
    assert action == "BUY" and 0 < conf <= 100


def test_meanrev_holds_without_dip():
    closes = [100.0 + i * 0.5 for i in range(120)]
    fn = make_meanrev_signal_fn()
    assert fn("A", _df(closes), {"mr_window": 10, "z_entry": 1.0})[0] == "HOLD"


def test_meanrev_trend_filter_blocks_downtrend_dip():
    closes = [200.0 - i * 0.8 for i in range(120)]  # 하락추세
    closes[-1] -= 10.0
    fn = make_meanrev_signal_fn()
    p = {"mr_window": 10, "z_entry": 1.0, "trend_window": 100}
    assert fn("A", _df(closes), dict(p, use_trend=1))[0] == "HOLD"
    assert fn("A", _df(closes), dict(p, use_trend=0))[0] == "BUY"


def test_dipbuy_requires_both_drop_and_volume_surge():
    closes = [100.0] * 30
    closes[-1] = 92.0  # -8%
    vols = [1000] * 30
    fn = make_dipbuy_signal_fn()
    p = {"drop_pct": 0.05, "vol_mult": 1.5}
    assert fn("A", _df(closes, vols), p)[0] == "HOLD"  # 거래량 평범 → HOLD
    vols[-1] = 5000
    action, conf = fn("A", _df(closes, vols), p)
    assert action == "BUY" and 50.0 < conf <= 100.0


def test_dipbuy_holds_on_small_drop():
    closes = [100.0] * 30
    closes[-1] = 98.0  # -2%
    vols = [1000] * 29 + [5000]
    fn = make_dipbuy_signal_fn()
    assert fn("A", _df(closes, vols), {"drop_pct": 0.05, "vol_mult": 1.5})[0] == "HOLD"


def test_entropy_damped_reduces_conf_only_when_enabled():
    choppy = [100.0]
    for i in range(40):
        choppy.append(choppy[-1] * (1.05 if i % 2 == 0 else 0.96))
    h = _df(choppy)
    assert entropy_damped(80.0, h, {}) == 80.0  # weight 미지정 → 무효과
    damped = entropy_damped(80.0, h, {"entropy_weight": 0.5, "entropy_window": 20})
    assert damped < 80.0


def test_signals_hold_on_short_history():
    fn_m, fn_d = make_meanrev_signal_fn(), make_dipbuy_signal_fn()
    short = _df([100.0] * 5)
    assert fn_m("A", short, {})[0] == "HOLD"
    assert fn_d("A", short, {})[0] == "HOLD"
