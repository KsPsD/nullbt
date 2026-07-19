"""블록 부트스트랩 성과 신뢰구간.

point estimate은 과신을 부른다. Sharpe 1.2 하나만 보면 그게 얼마나 흔들리는 값인지
알 수 없다. 같은 전략의 일별 수익률을 블록 단위로 재표본해서(자기상관·변동성 군집 보존)
성과 지표의 표본 변동성을 신뢰구간으로 낸다.

★한계(정직하게): 이건 '관측된 수익 스트림'의 재배열 변동성만 잰다. 새로운 시장 국면이나
다른 유니버스를 만들어내지 못한다 — 그건 더 무거운 도구(합성 경로 생성)의 몫이다.
플라시보가 '신호를 셔플'한다면, 이건 '수익 스트림을 블록 단위로 재표본'한다.

참고: Politis & Romano (1994), stationary bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nullbt.metrics import cagr, max_drawdown, sharpe_ratio


@dataclass
class MetricCI:
    """단일 지표의 점추정 + 부트스트랩 신뢰구간."""

    point: float
    low: float
    high: float
    mean: float

    def __str__(self) -> str:
        return f"{self.point:.3f}  (CI {self.low:.3f} ~ {self.high:.3f})"


@dataclass
class BootstrapResult:
    sharpe: MetricCI
    cagr: MetricCI
    max_drawdown: MetricCI
    n_resamples: int
    expected_block: int
    ci_level: float
    samples: dict  # 지표명 -> 재표본값 ndarray (히스토그램/추가 분석용)

    def format_report(self) -> str:
        pct = int(round(self.ci_level * 100))
        return (
            f"블록 부트스트랩 ({self.n_resamples}회, 기대 블록 {self.expected_block}일, "
            f"{pct}% CI)\n"
            f"  Sharpe : {self.sharpe}\n"
            f"  CAGR   : {self.cagr}\n"
            f"  MaxDD  : {self.max_drawdown}"
        )


def _stationary_indices(
    n: int, expected_block: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap 인덱스.

    기하분포(평균=expected_block) 길이의 블록을 임의 시작점에서 뽑아 순환(circular)
    래핑하며 n개를 채운다. 블록 길이를 랜덤화해 특정 블록 경계에 대한 민감도를 없애고,
    블록 내부는 원 시계열의 자기상관/변동성 군집을 보존한다.
    """
    p = 1.0 / expected_block
    idx = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))  # >= 1
        for j in range(length):
            if filled >= n:
                break
            idx[filled] = (start + j) % n
            filled += 1
    return idx


def _metrics_from_returns(
    r: pd.Series, periods_per_year: int
) -> tuple[float, float, float]:
    equity = (1.0 + r).cumprod()
    return (
        sharpe_ratio(r, periods_per_year),
        cagr(equity, periods_per_year),
        max_drawdown(equity),
    )


def bootstrap_metrics(
    daily_returns,
    n_resamples: int = 1000,
    expected_block: int = 20,
    periods_per_year: int = 252,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """일별 수익률의 블록 부트스트랩으로 Sharpe/CAGR/MaxDD 신뢰구간을 낸다.

    Parameters
    ----------
    daily_returns : 일별 수익률(list/Series/ndarray). NaN 제거. 보통
        ``result.equity.pct_change().dropna()`` 를 넣는다.
    n_resamples : 재표본 횟수.
    expected_block : 기대 블록 길이(일). 자기상관 척도. 클수록 군집을 더 보존.
    periods_per_year : 연율화 기준(일봉이면 252).
    ci_level : 신뢰수준(0~1). 0.95면 2.5~97.5 퍼센타일.
    seed : 재현용 시드.

    Returns
    -------
    BootstrapResult
    """
    r = pd.Series(daily_returns, dtype="float64").dropna().reset_index(drop=True)
    n = len(r)
    if n < 2:
        raise ValueError("daily_returns가 너무 짧다 (>= 2 필요).")
    if not 1 <= expected_block <= n:
        raise ValueError(f"expected_block은 1~{n} 범위여야 한다 (받음: {expected_block}).")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level은 (0, 1) 범위여야 한다 (받음: {ci_level}).")
    if n_resamples < 1:
        raise ValueError("n_resamples는 >= 1 이어야 한다.")

    rng = np.random.default_rng(seed)
    r_arr = r.to_numpy()
    sh = np.empty(n_resamples)
    cg = np.empty(n_resamples)
    md = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = _stationary_indices(n, expected_block, rng)
        sh[k], cg[k], md[k] = _metrics_from_returns(pd.Series(r_arr[idx]), periods_per_year)

    point_sh, point_cg, point_md = _metrics_from_returns(r, periods_per_year)
    alpha = (1.0 - ci_level) / 2.0

    def _ci(point: float, arr: np.ndarray) -> MetricCI:
        lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
        return MetricCI(point=float(point), low=float(lo), high=float(hi), mean=float(arr.mean()))

    return BootstrapResult(
        sharpe=_ci(point_sh, sh),
        cagr=_ci(point_cg, cg),
        max_drawdown=_ci(point_md, md),
        n_resamples=n_resamples,
        expected_block=expected_block,
        ci_level=ci_level,
        samples={"sharpe": sh, "cagr": cg, "max_drawdown": md},
    )
