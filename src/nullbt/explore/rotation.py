"""로테이션(포트폴리오) 전략 탐색 — 기존 explorer와 동일한 정직성 프로토콜.

engine.run_backtest가 아니라 rotation.run_rotation_backtest를 쓰기 때문에
search/report의 시그니처(signal_fn 고정)를 재사용할 수 없어 얇은 글루를 별도 구현.
프로토콜은 동일: in-sample 한정 탐색 → 교차-trial sharpe_std → IS 기반 DSR 랭킹
(OOS/holdout은 진단용). 랭킹/포맷은 report.rank_candidates/format_report 재사용.
"""

import math
import statistics

import optuna

from nullbt.engine import BacktestConfig
from nullbt.metrics import deflated_sharpe, max_drawdown, sharpe_ratio, win_rate
from nullbt.rotation import RankFn, run_rotation_backtest
from nullbt.validation import split_three_way, walk_forward_windows
from nullbt.explore.report import format_report, rank_candidates
from nullbt.explore.spec import StrategySpec, penalty, sample_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _measure_on(price_data, dates, rank_fn, params, config):
    res = run_rotation_backtest(price_data, list(dates), rank_fn, params, config)
    rets = res.equity.pct_change().dropna()
    return sharpe_ratio(rets), max_drawdown(res.equity), res.trades, len(rets)


def run_rotation_exploration(
    price_data: dict,
    rank_fn: RankFn,
    spec: StrategySpec,
    n_trials: int = 100,
    n_folds: int = 3,
    top_k: int = 5,
    config: BacktestConfig | None = None,
    dsr_trials: int | None = None,
) -> str:
    config = config or BacktestConfig()
    all_dates = sorted({d for df in price_data.values() for d in df.index})
    train, test, holdout = split_three_way(all_dates)
    windows = walk_forward_windows(train, n_folds)

    def objective(trial):
        params = sample_params(trial, spec.search_space)
        sharpes, mdds, pnls, total = [], [], [], 0
        for win in windows:
            s, m, trades, _ = _measure_on(price_data, win, rank_fn, params, config)
            sharpes.append(s)
            mdds.append(m)
            pnls.extend(t.pnl_pct for t in trades)
            total += len(trades)
        metrics = {
            "sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
            "mdd": max(mdds) if mdds else 0.0,
            "trades": total,
            "win_rate": win_rate(pnls),
        }
        trial.set_user_attr("is_sharpe", metrics["sharpe"])
        return metrics["sharpe"] - penalty(metrics, spec.constraints)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.value is not None]
    trial_sharpes = [t.user_attrs.get("is_sharpe", 0.0) for t in completed]
    if len(trial_sharpes) >= 2:
        sharpe_std = statistics.stdev(trial_sharpes) / math.sqrt(252)
        if sharpe_std <= 0:
            sharpe_std = 1.0
    else:
        sharpe_std = 1.0

    is_dates = [d for w in windows for d in w]
    trials = sorted(study.trials, key=lambda t: (t.value if t.value is not None else -1e9),
                    reverse=True)
    candidates = []
    for t in trials[:top_k]:
        is_sharpe, is_mdd, is_trades, n_obs = _measure_on(
            price_data, is_dates, rank_fn, t.params, config)
        oos_sharpe, _, _, _ = _measure_on(price_data, test, rank_fn, t.params, config)
        ho_sharpe, _, _, _ = _measure_on(price_data, holdout, rank_fn, t.params, config)
        dsr = deflated_sharpe(is_sharpe / math.sqrt(252), dsr_trials or n_trials, n_obs,
                              sharpe_std=sharpe_std)
        candidates.append({
            "params": t.params,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "holdout_sharpe": ho_sharpe,
            "mdd": is_mdd,
            "trades": len(is_trades),
            "overfit_gap": is_sharpe - oos_sharpe,
            "dsr": dsr,
        })
    return format_report(rank_candidates(candidates))
