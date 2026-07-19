"""기존 BreakoutMomentum 전략을 백테스트 엔진의 signal_fn으로 감싸는 어댑터.

기존 전략/분석기 코드는 수정하지 않고 재사용만 한다 (daily-only 모드).
"""

import pandas as pd

from nullbt.indicators import TechnicalAnalyzer
from nullbt.engine import SignalFn
from nullbt.entropy import return_entropy
from nullbt.examples.breakout import BreakoutMomentumStrategy

_MIN_HISTORY = 20


def make_breakout_signal_fn() -> SignalFn:
    analyzer = TechnicalAnalyzer()

    def signal_fn(code: str, history: pd.DataFrame, params: dict) -> tuple[str, float]:
        if history is None or len(history) < _MIN_HISTORY:
            return ("HOLD", 0.0)
        config = {
            "volume_threshold": params.get("volume_threshold", 1.5),
            "confidence_threshold": params.get("confidence_threshold", 70),
        }
        current_price = {
            "price": float(history["close"].iloc[-1]),
            "volume": int(history["volume"].iloc[-1]),
        }
        daily_technical = analyzer.analyze_daily(history, current_price["price"])
        data = {
            "stock_code": code,
            "current_price": current_price,
            "ohlcv": history,
            "daily_technical": daily_technical,
        }
        signal = BreakoutMomentumStrategy(config).generate_signal(data)
        action = signal.get("action", "HOLD")
        conf = float(signal.get("confidence", 0))

        # 정보이론 레짐 필터(연구용, 탐색 가능): 수익률 엔트로피가 높은(choppy) 구간일수록
        # 확신을 감쇠. entropy_weight=0(기본)이면 아무 영향 없음 → 기존 동작과 동일.
        # 탐색기가 이 피처가 실제로 도움이 되는지 정직하게 판정하도록 파라미터로 노출한다.
        w = float(params.get("entropy_weight", 0.0))
        if w > 0.0 and action in ("BUY", "STRONG_BUY"):
            ent = return_entropy(
                history["close"],
                window=int(params.get("entropy_window", 20)),
            )
            conf = conf * (1.0 - w * ent)  # ent∈[0,1], w∈[0,~0.5] → 감쇠만(증폭 없음)

        return (action, conf)

    return signal_fn
