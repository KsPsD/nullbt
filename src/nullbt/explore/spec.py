"""전략탐색 spec: 목적함수 + 제약 + 탐색공간."""

from dataclasses import dataclass


@dataclass
class StrategySpec:
    constraints: dict
    search_space: dict
    objective: str = "sharpe"


def penalty(metrics: dict, constraints: dict) -> float:
    """제약 위반량(≥0). 클수록 나쁨. objective에서 감점으로 사용."""
    p = 0.0
    max_mdd = constraints.get("max_mdd")
    if max_mdd is not None and metrics.get("mdd", 0.0) > max_mdd:
        p += (metrics["mdd"] - max_mdd) * 10.0  # 초과 비율 비례
    min_trades = constraints.get("min_trades")
    if min_trades is not None and metrics.get("trades", 0) < min_trades:
        deficit = (min_trades - metrics.get("trades", 0)) / min_trades
        p += deficit * 100.0  # 과소표본은 강한 페널티
    min_wr = constraints.get("min_win_rate")
    if min_wr is not None and metrics.get("win_rate", 0.0) < min_wr:
        p += (min_wr - metrics["win_rate"]) * 10.0
    return p


def sample_params(trial, search_space: dict) -> dict:
    """탐색공간 → Optuna trial 샘플. (int,int)→suggest_int, (float,float)→suggest_float."""
    params = {}
    for name, rng in search_space.items():
        lo, hi = rng
        if isinstance(lo, int) and isinstance(hi, int):
            params[name] = trial.suggest_int(name, lo, hi)
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi))
    return params
