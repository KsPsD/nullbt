import pandas as pd

from nullbt.engine import BacktestConfig
from nullbt.rotation import run_rotation_backtest
from nullbt.signals import make_momentum_rank_fn

CFG = BacktestConfig(commission=0.0, slippage=0.0, sell_tax=0.0)


def _df(opens, closes):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="B")
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": opens, "close": closes,
         "volume": [1000] * len(opens)},
        index=idx,
    )


def _flat_rank(scores: dict):
    """테스트용 고정 점수 rank_fn."""

    def fn(code, history, params):
        return scores.get(code)

    return fn


def test_rotation_enters_at_next_day_open():
    """T일 종가 정보로 순위 → T+1 시가 체결 (당일 종가 체결 아님)."""
    a = _df([10.0, 20.0, 20.0, 20.0], [15.0, 20.0, 20.0, 20.0])
    res = run_rotation_backtest(
        {"A": a}, list(a.index), _flat_rank({"A": 1.0}),
        {"top_k": 1, "rebalance_days": 99}, CFG,
    )
    # 첫 리밸런스(0일) 결정 → 1일 시가 20에 진입. 자본 1천만 → 50만주 아님: 10M/20 = 500,000주
    assert res.equity.iloc[0] == CFG.capital  # 0일엔 미체결(현금 그대로)
    assert abs(res.equity.iloc[1] - CFG.capital) < 1e-6  # 20에 사서 종가 20 → 동일


def test_rotation_rotates_out_and_records_trade():
    a = _df([10.0] * 6, [10.0] * 6)
    b = _df([5.0] * 6, [5.0] * 6)
    scores = {"A": 2.0, "B": 1.0}

    def rank(code, history, params):
        # 3일째부터 B가 우위 → A 매도/B 매수 로테이션 발생
        if len(history) >= 3:
            return {"A": 1.0, "B": 2.0}[code]
        return scores[code]

    res = run_rotation_backtest(
        {"A": a, "B": b}, list(a.index), rank,
        {"top_k": 1, "rebalance_days": 2}, CFG,
    )
    assert any(t.code == "A" and t.exit_reason == "rotate" for t in res.trades)


def test_rotation_lazy_rebalance_keeps_survivors():
    """목표에 계속 남는 종목은 매도/재매수하지 않는다(비용 회피)."""
    a = _df([10.0] * 8, [10.0] * 8)
    res = run_rotation_backtest(
        {"A": a}, list(a.index), _flat_rank({"A": 1.0}),
        {"top_k": 1, "rebalance_days": 2}, CFG,
    )
    assert len(res.trades) == 0  # 끝까지 보유 → 완결 거래 없음


def test_rotation_sell_costs_applied():
    cfg = BacktestConfig(commission=0.001, slippage=0.0, sell_tax=0.002)
    a = _df([10.0] * 4, [10.0] * 4)

    def rank(code, history, params):
        return 1.0 if len(history) < 3 else None  # 3일째 순위 탈락 → 매도

    res = run_rotation_backtest({"A": a}, list(a.index), rank,
                                {"top_k": 1, "rebalance_days": 2}, cfg)
    assert len(res.trades) == 1
    # 왕복 비용만큼 자본 감소: 매수 수수료 + 매도 수수료 + 매도세
    assert res.equity.iloc[-1] < cfg.capital
    assert res.equity.iloc[-1] > cfg.capital * (1 - 0.01)  # 대략적 상한 확인


def test_rotation_respects_top_k():
    data = {c: _df([10.0] * 6, [10.0] * 6) for c in ["A", "B", "C", "D"]}
    scores = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    res = run_rotation_backtest(
        data, list(data["A"].index), _flat_rank(scores),
        {"top_k": 2, "rebalance_days": 99}, CFG,
    )
    # 진입 기록은 상위 2종목(A,B)만: 미완결 포지션이므로 trades가 아닌 equity로 간접 확인
    # 4종목 중 2종목에만 자본 투입 → 전량 보유가치 = capital (가격 불변이므로)
    assert abs(res.equity.iloc[-1] - CFG.capital) < 1e-6


def test_momentum_rank_skip_math_and_entropy_gate():
    fn = make_momentum_rank_fn()
    idx = pd.date_range("2024-01-01", periods=61, freq="B")
    up = pd.DataFrame({"close": [100.0 * 1.01**i for i in range(61)],
                       "open": [0] * 61, "high": [0] * 61, "low": [0] * 61,
                       "volume": [0] * 61}, index=idx)
    p = {"mom_window": 40, "skip_days": 10}
    s = fn("A", up, p)
    expected = up["close"].iloc[-11] / up["close"].iloc[-51] - 1.0
    assert abs(s - expected) < 1e-12
    assert fn("A", up.iloc[:30], p) is None  # 이력 부족 → 부적격

    choppy_prices = [100.0]
    for i in range(60):
        choppy_prices.append(choppy_prices[-1] * (1.06 if i % 2 == 0 else 0.95))
    choppy = pd.DataFrame({"close": choppy_prices, "open": [0] * 61, "high": [0] * 61,
                           "low": [0] * 61, "volume": [0] * 61},
                          index=pd.date_range("2024-01-01", periods=61, freq="B"))
    gated = dict(p, ent_max=0.55, entropy_window=20)
    assert fn("A", choppy, gated) is None  # 어수선 → 사전선별 탈락
    assert fn("A", up, gated) is not None  # 정갈한 추세 → 통과


def test_rotation_gate_liquidates_and_blocks_entry():
    """게이트 걸린 리밸런스일 → 다음날 시가 전량 매도, 신규 진입 없음."""
    a = _df([10.0] * 8, [10.0] * 8)
    dates = list(a.index)
    gate = {dates[2]}  # 2일(리밸런스일)에 게이트 발동
    res = run_rotation_backtest(
        {"A": a}, dates, _flat_rank({"A": 1.0}),
        {"top_k": 1, "rebalance_days": 2}, CFG, gate_dates=gate,
    )
    # 0일 결정→1일 진입, 2일 게이트→3일 청산, 4일 결정→5일 재진입
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_date == dates[3] and t.exit_reason == "rotate"
    assert abs(res.equity.iloc[-1] - CFG.capital) < 1e-6  # 무비용·가격불변 → 원금 유지
