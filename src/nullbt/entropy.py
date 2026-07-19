"""정보이론 기반 레짐 피처 (순수 함수, 네트워크/무거운 의존성 없음).

수익률 분포의 Shannon 엔트로피로 '레짐의 무질서도'를 측정한다.
- 낮은 엔트로피 = 수익률이 소수의 구간에 집중 = 정갈한 추세/레짐 (모멘텀 전략에 유리)
- 높은 엔트로피 = 수익률이 넓게 분산 = choppy/무작위 (모멘텀 전략에 불리)

정규화 엔트로피 H_norm ∈ [0, 1] 을 신뢰도 감쇠 필터로 쓴다.
(참고: Shannon 1948; 시장 레짐/예측가능성에 정보엔트로피를 적용하는 계열)
"""

import math

import pandas as pd


def normalized_entropy(values, bins: int = 3) -> float:
    """값 분포의 정규화 Shannon 엔트로피 ∈ [0, 1].

    값을 [min, max]의 등폭 `bins`개 구간으로 이산화한 뒤 H = -Σ p·ln p 를 ln(bins)로 정규화.
    - 모두 동일(분산 0) → 0.0 (완전한 질서)
    - 구간에 균등 분포 → 1.0 (완전한 무질서)
    NaN은 무시. 표본 < 2 또는 bins < 2 이면 0.0.
    """
    xs = [float(v) for v in values if v == v]  # NaN 제외 (NaN != NaN)
    if len(xs) < 2 or bins < 2:
        return 0.0
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return 0.0  # 모든 값 동일 → 무질서 없음
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        idx = int((x - lo) / width)
        if idx >= bins:  # 최댓값 경계 보정
            idx = bins - 1
        counts[idx] += 1
    n = len(xs)
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log(p)
    return float(h / math.log(bins))


def return_entropy(prices, window: int = 20, eps: float = 0.0) -> float:
    """최근 `window`일 일간수익률의 '방향' 엔트로피 ∈ [0, 1].

    수익률을 부호로 상승/보합/하락 3범주로 분류한 뒤 Shannon 엔트로피를 ln(3)로 정규화.
    - 방향이 일관(예: 계속 상승) → 0.0 (정갈한 추세)
    - 방향이 뒤섞임(등락 반복) → 높음 (choppy)
    prices는 종가 시퀀스. eps는 보합 판정 임계(기본 0).

    주의: 크기가 아니라 '방향의 무질서'를 측정한다(모멘텀 레짐 필터 목적). 크기 분산이
    필요하면 normalized_entropy()를 직접 쓸 것. 두 방향만 나타나면 최댓값은 ln2/ln3≈0.63.
    """
    s = pd.Series(list(prices), dtype="float64")
    rets = s.pct_change().dropna().tail(window)
    if len(rets) < 2:
        return 0.0
    up = int((rets > eps).sum())
    down = int((rets < -eps).sum())
    flat = len(rets) - up - down
    n = len(rets)
    h = 0.0
    for c in (up, down, flat):
        if c:
            p = c / n
            h -= p * math.log(p)
    return float(h / math.log(3))  # 가능한 범주 수(상승/보합/하락)=3으로 정규화
