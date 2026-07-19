"""이벤트 드리븐 일봉 백테스트 엔진. 전략과 분리(signal_fn 주입)."""

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

SignalFn = Callable[[str, pd.DataFrame, dict], tuple[str, float]]


@dataclass
class BacktestConfig:
    commission: float = 0.00015
    slippage: float = 0.002
    # 한국 매도분 증권거래세(+농특세). 매도 체결에만 적용(매수엔 미적용).
    # 대표값 0.18%. 연도/시장(KOSPI/KOSDAQ)별로 상이하니 필요시 조정.
    sell_tax: float = 0.0018
    max_positions: int = 10
    capital: float = 10_000_000.0


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list = field(default_factory=list)


@dataclass
class _Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    bars_held: int = 0


def run_backtest(
    price_data: dict[str, pd.DataFrame],
    dates: list,
    signal_fn: SignalFn,
    params: dict,
    config: BacktestConfig,
) -> BacktestResult:
    cutoff = params["confidence_cutoff"]
    stop = params["stop_pct"]
    target = params["target_pct"]
    max_hold = params["max_hold_days"]

    cash = config.capital
    size = config.capital / config.max_positions
    positions: dict[str, _Position] = {}
    pending: list[str] = []
    trades: list[Trade] = []
    equity_curve: dict = {}

    for date in dates:
        # 1) 대기 진입을 오늘 시가에 체결
        for code in pending:
            if code in positions or len(positions) >= config.max_positions:
                continue
            df = price_data.get(code)
            if df is None or date not in df.index:
                continue
            open_px = df.loc[date, "open"] * (1 + config.slippage)
            if open_px <= 0:
                continue
            shares = size / open_px
            cash -= shares * open_px * (1 + config.commission)
            positions[code] = _Position(code, date, open_px, shares)
        pending = []

        # 2) 보유 포지션 청산 점검 (손절 우선 → 목표 → 최대보유일)
        for code in list(positions.keys()):
            pos = positions[code]
            if date == pos.entry_date:
                continue  # 진입 당일은 청산 안 함
            df = price_data.get(code)
            if df is None or date not in df.index:
                continue
            row = df.loc[date]
            pos.bars_held += 1
            stop_px = round(pos.entry_price * (1 - stop), 10)
            target_px = round(pos.entry_price * (1 + target), 10)
            exit_px = None
            reason = None
            if row["low"] <= stop_px:
                # 갭 하락으로 시가가 손절가보다 낮으면 손절가가 아니라 시가(더 나쁜 가격)에
                # 체결 — 갭 관통을 무시하면 손실이 과소계상됨(KNOWN_LIMITATIONS.md #8).
                exit_px, reason = min(stop_px, row["open"]), "stop"
            elif row["high"] >= target_px:
                # 목표가는 보수적으로 target_px 유지(갭 상승 시 실제로는 더 높게 체결될 수 있으나
                # 이익을 과대계상하지 않도록 낮게 잡음).
                exit_px, reason = target_px, "target"
            elif pos.bars_held >= max_hold:
                exit_px, reason = row["close"], "max_hold"
            if exit_px is not None:
                fill = exit_px * (1 - config.slippage)
                cash += pos.shares * fill * (1 - config.commission) * (1 - config.sell_tax)
                trades.append(
                    Trade(code, pos.entry_date, pos.entry_price, date, fill,
                          fill / pos.entry_price - 1, reason)
                )
                del positions[code]

        # 3) 종가 기준 자산 평가
        mtm = cash
        for code, pos in positions.items():
            df = price_data.get(code)
            if df is not None and date in df.index:
                mtm += pos.shares * df.loc[date, "close"]
            else:
                mtm += pos.shares * pos.entry_price
        equity_curve[date] = mtm

        # 4) ≤T 데이터로 시그널 생성 → 다음날 시가 진입 큐잉
        if len(positions) < config.max_positions:
            cands: list[tuple[float, str]] = []
            for code, df in price_data.items():
                if code in positions or date not in df.index:
                    continue
                hist = df.loc[:date]  # ★ T까지만 — lookahead 차단
                action, conf = signal_fn(code, hist, params)
                if action in ("BUY", "STRONG_BUY") and conf >= cutoff:
                    cands.append((conf, code))
            cands.sort(reverse=True)
            slots = config.max_positions - len(positions)
            pending = [code for _, code in cands[:slots]]

    # 5) 백테스트 종료 시 잔여 포지션 강제 청산 (마지막 종가)
    last_date = dates[-1] if dates else None
    if last_date is not None:
        for code in list(positions.keys()):
            pos = positions[code]
            df = price_data.get(code)
            if df is not None and last_date in df.index:
                close_px = df.loc[last_date, "close"]
            else:
                close_px = pos.entry_price
            fill = close_px * (1 - config.slippage)
            cash += pos.shares * fill * (1 - config.commission) * (1 - config.sell_tax)
            trades.append(
                Trade(code, pos.entry_date, pos.entry_price, last_date, fill,
                      fill / pos.entry_price - 1, "eob")
            )
        positions.clear()

    return BacktestResult(equity=pd.Series(equity_curve), trades=trades)
