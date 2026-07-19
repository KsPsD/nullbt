"""Optuna 기반 inverse-design 탐색. objective는 in-sample 윈도우에서만 계산(누수 차단)."""

import optuna

from nullbt.engine import BacktestConfig, run_backtest
from nullbt.metrics import max_drawdown, sharpe_ratio, win_rate
from nullbt.explore.spec import StrategySpec, penalty, sample_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _aggregate(price_data, windows, signal_fn, params, config) -> dict:
    """in-sample 윈도우별 백테스트 → 집계 지표."""
    sharpes, mdds, pnls = [], [], []
    total_trades = 0
    for win in windows:
        res = run_backtest(price_data, win, signal_fn, params, config)
        rets = res.equity.pct_change().dropna()
        sharpes.append(sharpe_ratio(rets))
        mdds.append(max_drawdown(res.equity))
        pnls.extend(t.pnl_pct for t in res.trades)
        total_trades += len(res.trades)
    mean_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    worst_mdd = max(mdds) if mdds else 0.0
    return {
        "sharpe": mean_sharpe,
        "mdd": worst_mdd,
        "trades": total_trades,
        "win_rate": win_rate(pnls),
    }


def evaluate_params(
    params: dict, windows, price_data, signal_fn, spec: StrategySpec, config: BacktestConfig
) -> tuple[float, dict]:
    """파라미터 1세트 → (score, 집계지표). score = objective - penalty."""
    metrics = _aggregate(price_data, windows, signal_fn, params, config)
    score = metrics.get(spec.objective, 0.0) - penalty(metrics, spec.constraints)
    return score, metrics


def run_search(
    spec: StrategySpec, price_data, windows, signal_fn, config: BacktestConfig, n_trials: int
) -> optuna.Study:
    def objective(trial):
        params = sample_params(trial, spec.search_space)
        score, metrics = evaluate_params(params, windows, price_data, signal_fn, spec, config)
        # DSR의 sharpe_std(교차-trial Sharpe 분산) 산출용으로 trial별 IS Sharpe 보존.
        trial.set_user_attr("is_sharpe", metrics.get("sharpe", 0.0))
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study
