import pandas as pd

from nullbt.engine import BacktestConfig, run_backtest


def _df(prices):
    """종가=고가=저가=시가=prices, 거래량 1000 고정인 일봉 df."""
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": [1000] * len(prices)},
        index=idx,
    )


def _always_buy(code, history, params):
    return ("BUY", 100.0)


def _never_buy(code, history, params):
    return ("HOLD", 0.0)


def test_no_trades_when_never_buy():
    data = {"A": _df([100, 101, 102, 103])}
    cfg = BacktestConfig(max_positions=1, capital=1_000_000)
    res = run_backtest(data, list(data["A"].index), _never_buy, _PARAMS, cfg)
    assert res.trades == []


def test_entry_fills_at_next_day_open_not_signal_day_close():
    # 신호는 day0 종가에 발생 → day1 시가에 체결되어야 함 (lookahead 금지)
    prices = [100, 200, 210, 220]  # day1 시가=200
    data = {"A": _df(prices)}
    cfg = BacktestConfig(max_positions=1, capital=1_000_000, slippage=0.0, commission=0.0)
    res = run_backtest(data, list(data["A"].index), _always_buy, _PARAMS, cfg)
    assert len(res.trades) >= 1
    first = res.trades[0]
    assert first.entry_price == 200  # day0 종가(100)가 아니라 day1 시가(200)
    assert first.entry_date == data["A"].index[1]


def test_max_positions_capped():
    data = {c: _df([100, 101, 102, 103]) for c in ["A", "B", "C"]}
    dates = list(data["A"].index)
    cfg = BacktestConfig(max_positions=2, capital=1_000_000, slippage=0.0, commission=0.0)
    res = run_backtest(data, dates, _always_buy, _PARAMS, cfg)
    # 한 시점 동시보유 ≤ 2 (진입은 max 2종목까지만 큐잉)
    open_dates = [t.entry_date for t in res.trades]
    # 같은 날 진입한 종목 수 ≤ max_positions
    from collections import Counter
    assert max(Counter(open_dates).values()) <= 2


_PARAMS = {"confidence_cutoff": 20.0, "stop_pct": 0.05, "target_pct": 0.10, "max_hold_days": 5}


def test_stop_fills_at_gap_open_when_below_stop():
    # day1 시가=100 진입, stop_pct=5% → 손절가=95.
    # day2가 갭 하락해 시가 90(<95)로 열리면 손절가(95)가 아니라 시가(90)에 체결되어야 함.
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {"open": [100, 100, 90], "high": [100, 100, 91],
         "low": [100, 100, 88], "close": [100, 100, 89], "volume": [1000] * 3},
        index=idx,
    )
    cfg = BacktestConfig(max_positions=1, capital=1_000_000, slippage=0.0, commission=0.0, sell_tax=0.0)
    res = run_backtest({"A": df}, list(idx), _always_buy, _PARAMS, cfg)
    stops = [t for t in res.trades if t.exit_reason == "stop"]
    assert len(stops) == 1
    assert stops[0].exit_price == 90  # 갭 관통 → 시가 체결(손실 과소계상 방지), 95 아님


def test_sell_tax_reduces_proceeds():
    # 목표가 체결(110) 시 매도세금이 최종 자산을 낮춰야 함.
    prices = [100, 100, 120]  # day1 시가=100 진입, day2 고가=120 → 목표가 110 체결
    data = {"A": _df(prices)}
    dates = list(data["A"].index)
    base = dict(max_positions=1, capital=1_000_000, slippage=0.0, commission=0.0)
    r0 = run_backtest(data, dates, _always_buy, _PARAMS, BacktestConfig(sell_tax=0.0, **base))
    r1 = run_backtest(data, dates, _always_buy, _PARAMS, BacktestConfig(sell_tax=0.1, **base))
    assert r1.equity.iloc[-1] < r0.equity.iloc[-1]
