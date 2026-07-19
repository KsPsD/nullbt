"""횡단면 로테이션 백테스트 (포트폴리오 방식).

기존 engine.run_backtest(종목별 신호 + 고정 손절/목표, 단기 보유)와 달리,
리밸런스일마다 순위(rank_fn) 상위 top_k 종목을 보유한다. 왕복 비용이 높은
KR 시장에서 회전을 낮추는 전략군을 테스트하기 위한 엔진.

규율은 기존 엔진과 동일:
- T일 종가까지의 정보로 순위 산출 → T+1일 시가 체결 (lookahead 차단)
- 수수료/슬리피지 양방향, 매도세는 매도에만
- lazy rebalance: 목표에 계속 남는 종목은 재조정하지 않음(불필요한 왕복 비용 회피).
  신규 편입만 현금을 균등 분할해 매수, 탈락 종목만 매도.

rank_fn(code, history, params) -> float | None:
  None = 부적격(데이터 부족, 사전선별 탈락 등). 클수록 우선 편입.
"""

from collections.abc import Callable

import pandas as pd

from nullbt.engine import BacktestConfig, BacktestResult, Trade

RankFn = Callable[[str, pd.DataFrame, dict], float | None]


def run_rotation_backtest(
    price_data: dict[str, pd.DataFrame],
    dates: list,
    rank_fn: RankFn,
    params: dict,
    config: BacktestConfig,
    gate_dates: set | None = None,
) -> BacktestResult:
    """gate_dates: 절대모멘텀 등 시장 게이트가 발동된 날짜 집합(선택).
    리밸런스일이 게이트에 걸리면 목표를 빈 집합으로 → 다음날 시가 전량 매도 후
    현금 대기. 게이트가 풀린 다음 리밸런스일에 재진입한다."""
    top_k = int(params["top_k"])
    reb = max(1, int(params["rebalance_days"]))

    cash = config.capital
    shares: dict[str, float] = {}
    entry: dict[str, tuple] = {}  # code -> (entry_date, entry_price)
    last_px: dict[str, float] = {}
    trades: list[Trade] = []
    equity: dict = {}
    pending: set | None = None

    for i, date in enumerate(dates):
        # 1) 전일(리밸런스일) 종가 정보로 결정된 목표 포트폴리오를 오늘 시가에 체결
        if pending is not None:
            for code in list(shares):  # 탈락 종목 매도
                if code in pending:
                    continue
                df = price_data.get(code)
                if df is None or date not in df.index:
                    continue  # 오늘 거래 불가(정지 등) → 보유 유지, 다음 기회에 매도
                fill = float(df.loc[date, "open"]) * (1 - config.slippage)
                if fill <= 0:
                    continue
                cash += shares[code] * fill * (1 - config.commission) * (1 - config.sell_tax)
                e_date, e_px = entry.pop(code)
                trades.append(Trade(code, e_date, e_px, date, fill, fill / e_px - 1, "rotate"))
                del shares[code]
            buyable = []
            for code in sorted(pending):  # 신규 편입 매수(현금 균등 분할)
                if code in shares:
                    continue
                df = price_data.get(code)
                if df is not None and date in df.index and float(df.loc[date, "open"]) > 0:
                    buyable.append(code)
            if buyable and cash > 0:
                budget = cash / len(buyable)
                for code in buyable:
                    fill = float(price_data[code].loc[date, "open"]) * (1 + config.slippage)
                    n = budget / (fill * (1 + config.commission))
                    cash -= n * fill * (1 + config.commission)
                    shares[code] = n
                    entry[code] = (date, fill)
            pending = None

        # 2) 리밸런스일: 오늘 종가까지의 정보로 다음 목표 결정(체결은 내일 시가)
        if i % reb == 0:
            if gate_dates is not None and date in gate_dates:
                pending = set()  # 시장 게이트 → 전량 청산 목표(현금 대피)
            else:
                scores: dict[str, float] = {}
                for code, df in price_data.items():
                    if date not in df.index:
                        continue  # 오늘 시세 없는 종목은 순위 제외
                    s = rank_fn(code, df.loc[:date], params)
                    if s is not None and s == s:
                        scores[code] = float(s)
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                pending = {c for c, _ in ranked[:top_k]}

        # 3) 종가 기준 자산 평가 (시세 없는 날은 마지막 체결가 유지)
        mtm = cash
        for code, n in shares.items():
            df = price_data.get(code)
            if df is not None and date in df.index:
                last_px[code] = float(df.loc[date, "close"])
            mtm += n * last_px.get(code, entry[code][1])
        equity[date] = mtm

    return BacktestResult(equity=pd.Series(equity), trades=trades)
