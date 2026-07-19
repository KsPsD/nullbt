"""백테스트 성과 지표 (순수 함수)."""

import math

import pandas as pd
from scipy.stats import norm


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """일간 수익률 시리즈 → 연율화 Sharpe. 표준편차 0이면 0."""
    if returns is None or len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """자산곡선 → 최대 낙폭(양수 비율). 단조증가면 0."""
    if equity is None or len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(-drawdown.min())


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """자산곡선 → 연복리 수익률."""
    if equity is None or len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def win_rate(trade_pnls: list[float]) -> float:
    """거래별 손익률 리스트 → 승률."""
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def expected_max_sharpe(n_trials: int, sharpe_std: float = 1.0) -> float:
    """n_trials개 독립 시도의 Sharpe 기대 최댓값 (Bailey & López de Prado)."""
    if n_trials < 2:
        return 0.0
    euler = 0.5772156649015329
    e = math.e
    term = (1 - euler) * norm.ppf(1 - 1.0 / n_trials) + euler * norm.ppf(
        1 - 1.0 / (n_trials * e)
    )
    return float(sharpe_std * term)


def deflated_sharpe(
    observed_sharpe: float, n_trials: int, n_obs: int, sharpe_std: float = 1.0
) -> float:
    """관측 Sharpe를 시도 횟수로 할인한 DSR(확률, 0~1)."""
    if n_obs < 2:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sharpe_std)
    return float(norm.cdf((observed_sharpe - sr0) * math.sqrt(n_obs - 1)))
