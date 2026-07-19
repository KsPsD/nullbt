"""경량 리서치 시그널 라이브러리 (일봉 전용, 순수 함수).

기존 strategy_adapter(레거시 TechnicalAnalyzer 경유, 트라이얼당 수 분)와 달리
pandas 연산만 사용해 탐색을 빠르게 한다. 모든 시그널의 규약:
- 입력: (code, 당일 종가까지의 history DataFrame, params) — 미래 미참조
- 출력: ("BUY" | "HOLD", confidence 0~100)
- params에 entropy_weight>0이면 종목 단위 방향 엔트로피로 확신 감쇠
  (플라시보 대조로 검증된 필터 — docs/research/2026-07-02_entropy_regime_filter.md)

엔진이 요구하는 공통 params(confidence_cutoff, stop_pct, target_pct, max_hold_days)는
탐색공간에서 함께 샘플링된다(src/explorer/cli.py의 패밀리 스펙 참조).
"""

from nullbt.engine import SignalFn
from nullbt.entropy import return_entropy

_HOLD = ("HOLD", 0.0)


def entropy_damped(conf: float, history, params: dict) -> float:
    """검증된 종목 단위 엔트로피 감쇠. entropy_weight=0(기본)이면 무효과."""
    w = float(params.get("entropy_weight", 0.0))
    if w <= 0.0 or conf <= 0.0:
        return conf
    ent = return_entropy(history["close"], window=int(params.get("entropy_window", 20)))
    return conf * (1.0 - w * ent)


def make_meanrev_signal_fn() -> SignalFn:
    """단기 평균회귀: z-score 과매도 반등 매수(선택적 장기 추세 필터).

    z = (종가 - SMA(mr_window)) / STD(mr_window). z ≤ -z_entry일 때 매수 후보 —
    깊이 눌릴수록 확신 증가. use_trend=1이면 종가가 SMA(trend_window) 위일 때만
    (상승추세 속 눌림만 매수).
    """

    def fn(code: str, history, params: dict) -> tuple[str, float]:
        n = int(params.get("mr_window", 10))
        trend_n = int(params.get("trend_window", 100))
        use_trend = int(params.get("use_trend", 1))
        need = max(n, trend_n if use_trend else n) + 1
        if history is None or len(history) < need:
            return _HOLD
        close = history["close"]
        tail = close.tail(n)
        mu = float(tail.mean())
        sd = float(tail.std())
        if not sd or sd != sd or sd <= 0.0:
            return _HOLD
        z = (float(close.iloc[-1]) - mu) / sd
        z_entry = float(params.get("z_entry", 1.5))
        if z > -z_entry:
            return _HOLD
        if use_trend and float(close.iloc[-1]) < float(close.tail(trend_n).mean()):
            return _HOLD
        conf = min(100.0, 50.0 + 25.0 * (-z - z_entry))
        return ("BUY", entropy_damped(conf, history, params))

    return fn


def make_momentum_rank_fn():
    """횡단면 상대강도(RS) 순위 함수 — rotation 엔진용.

    점수 = skip_days 직전까지의 mom_window일 수익률(고전적 12-1 모멘텀의 일봉판:
    최근 skip_days는 단기 반전 소음이라 제외). 클수록 강한 종목.

    ent_max < 1이면 **엔트로피 사전선별**: 방향 엔트로피가 ent_max를 넘는(어수선한)
    종목은 순위에서 제외(None) — 검증된 종목 단위 필터를 감쇠가 아니라 선별자로 사용.
    """

    def fn(code: str, history, params: dict):
        w = int(params.get("mom_window", 120))
        skip = int(params.get("skip_days", 5))
        if history is None or len(history) < w + skip + 1:
            return None
        close = history["close"]
        past = float(close.iloc[-(w + skip + 1)])
        recent = float(close.iloc[-(skip + 1)]) if skip > 0 else float(close.iloc[-1])
        if past <= 0:
            return None
        ent_max = float(params.get("ent_max", 1.0))
        if ent_max < 1.0:
            ent = return_entropy(close, window=int(params.get("entropy_window", 20)))
            if ent > ent_max:
                return None
        return recent / past - 1.0

    return fn


def make_dipbuy_signal_fn() -> SignalFn:
    """투매 반등: 대폭 하락일 + 거래량 급증(투매 정점 후보) 다음날 반등 매수.

    당일 수익률 ≤ -drop_pct 이고 당일 거래량 ≥ vol_mult × 직전 20일 평균이면 매수 —
    낙폭이 클수록 확신 증가. 평균회귀와 달리 이벤트(단일 급락) 기반.
    """

    def fn(code: str, history, params: dict) -> tuple[str, float]:
        if history is None or len(history) < 22:
            return _HOLD
        close = history["close"]
        vol = history["volume"]
        prev = float(close.iloc[-2])
        if prev <= 0:
            return _HOLD
        ret1 = float(close.iloc[-1]) / prev - 1.0
        drop = float(params.get("drop_pct", 0.05))
        if ret1 > -drop:
            return _HOLD
        v_avg = float(vol.iloc[-21:-1].mean())
        if v_avg <= 0:
            return _HOLD
        if float(vol.iloc[-1]) < float(params.get("vol_mult", 1.5)) * v_avg:
            return _HOLD
        conf = min(100.0, 50.0 + 400.0 * (-ret1 - drop))
        return ("BUY", entropy_damped(conf, history, params))

    return fn
