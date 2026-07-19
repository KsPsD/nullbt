import numpy as np
import pandas as pd

from nullbt.examples.adapter import make_breakout_signal_fn


def _uptrend_df(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(100, 200, n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": [1000] * (n - 1) + [5000]},
        index=idx,
    )


def test_returns_action_and_confidence_tuple():
    fn = make_breakout_signal_fn()
    params = {"volume_threshold": 1.5, "confidence_threshold": 70}
    action, conf = fn("005930", _uptrend_df(), params)
    assert isinstance(action, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 100.0


def test_hold_when_insufficient_history():
    fn = make_breakout_signal_fn()
    short = _uptrend_df(10)
    action, conf = fn("005930", short, {"volume_threshold": 1.5, "confidence_threshold": 70})
    assert action == "HOLD"
    assert conf == 0.0


def _breakout_history():
    # test_cli의 검증된 BUY 패턴: 50일 플랫(100) → 마지막 봉 115로 돌파(거래량 급증),
    # 마지막 봉 high==close로 52주 신고가 조건 충족. 최근 수익률에 큰 점프 1개 → 엔트로피>0.
    close = [100.0] * 50 + [115.0]
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    df = pd.DataFrame(
        {"open": close, "high": [c * 1.02 for c in close], "low": [c * 0.98 for c in close],
         "close": close, "volume": [1000] * 50 + [5000]},
        index=idx,
    )
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"]
    return df


def test_entropy_weight_reduces_confidence_on_buy():
    fn = make_breakout_signal_fn()
    df = _breakout_history()
    base = {"volume_threshold": 1.5, "confidence_threshold": 20}
    base_action, base_conf = fn("X", df, base)
    w_action, w_conf = fn("X", df, {**base, "entropy_weight": 0.5, "entropy_window": 20})
    assert base_action in ("BUY", "STRONG_BUY")  # 신호가 실제로 발생
    assert w_conf < base_conf   # 엔트로피>0 구간에서 확신 감쇠
    assert w_action == base_action  # 감쇠는 confidence만, 액션은 유지


def test_entropy_weight_zero_is_noop():
    fn = make_breakout_signal_fn()
    df = _breakout_history()
    base = {"volume_threshold": 1.5, "confidence_threshold": 20}
    _, c0 = fn("X", df, base)
    _, c0b = fn("X", df, {**base, "entropy_weight": 0.0})
    assert c0 == c0b  # entropy_weight=0 → 기존 동작과 완전 동일


def test_consistency_with_direct_strategy_call():
    # 어댑터 결과 == 직접 전략 호출 결과 (백테스트=실거래 시그널 일치 보증)
    from nullbt.indicators import TechnicalAnalyzer
    from nullbt.examples.breakout import BreakoutMomentumStrategy

    df = _uptrend_df()
    cur = {"price": float(df["close"].iloc[-1]), "volume": int(df["volume"].iloc[-1])}
    daily = TechnicalAnalyzer().analyze_daily(df, cur["price"])
    direct = BreakoutMomentumStrategy({"volume_threshold": 1.5, "confidence_threshold": 70}).generate_signal(
        {"stock_code": "005930", "current_price": cur, "ohlcv": df, "daily_technical": daily}
    )
    fn = make_breakout_signal_fn()
    action, conf = fn("005930", df, {"volume_threshold": 1.5, "confidence_threshold": 70})
    assert action == direct["action"]
    assert conf == float(direct["confidence"])
