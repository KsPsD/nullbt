"""탐색 결과 리포트: IS/OOS/holdout 측정 + 오버핏 격차 순 정렬."""

import math

from nullbt.engine import BacktestConfig, run_backtest
from nullbt.metrics import deflated_sharpe, max_drawdown, sharpe_ratio


def _sharpe_on(price_data, dates, signal_fn, params, config) -> tuple[float, float, int, int]:
    res = run_backtest(price_data, list(dates), signal_fn, params, config)
    rets = res.equity.pct_change().dropna()
    return sharpe_ratio(rets), max_drawdown(res.equity), len(res.trades), len(rets)


def measure_candidate(
    params, is_windows, oos_dates, holdout_dates, price_data, signal_fn, config: BacktestConfig,
    n_trials: int = 1, sharpe_std: float = 1.0,
) -> dict:
    """후보 1개를 IS(전체)/OOS/holdout 구간에서 측정.

    sharpe_std: 교차-trial Sharpe 분산(per-period). DSR의 SR0 허들 산출에 사용.
    하드코딩 1.0 대신 실제 탐색에서 관측된 분산을 넘겨야 deflation이 보정됨.
    """
    is_dates = [d for w in is_windows for d in w]
    is_sharpe, is_mdd, is_trades, n_obs = _sharpe_on(price_data, is_dates, signal_fn, params, config)
    oos_sharpe, _, _, _ = _sharpe_on(price_data, oos_dates, signal_fn, params, config)
    holdout_sharpe, _, _, _ = _sharpe_on(price_data, holdout_dates, signal_fn, params, config)
    # De-annualize before passing to deflated_sharpe, which expects a per-period Sharpe.
    # sharpe_ratio() multiplies by sqrt(252); reverse that so the deflation is meaningful.
    dsr = deflated_sharpe(is_sharpe / math.sqrt(252), n_trials, n_obs, sharpe_std=sharpe_std)
    return {
        "params": params,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "holdout_sharpe": holdout_sharpe,
        "mdd": is_mdd,
        "trades": is_trades,
        "overfit_gap": is_sharpe - oos_sharpe,
        "dsr": dsr,
    }


def rank_candidates(
    candidates: list[dict], overfit_threshold: float = 1.0, dsr_threshold: float = 0.95
) -> list[dict]:
    """DSR(IS-only, 시도횟수 보정) 순 정렬 — 승자를 OOS/holdout으로 고르지 않는다.

    이전 구현은 oos_sharpe로 순위를 매겨(selection-on-validation) 보고된 OOS가 상향편향됐다.
    이제 선택 통계는 IS 기반 DSR이며, OOS/holdout/overfit_gap은 **진단용 리포트 값**일 뿐
    선택에 개입하지 않는다.
      - dsr_pass: DSR ≥ dsr_threshold(기본 0.95) 통과 여부
      - overfit: IS-OOS 격차 진단 플래그(참고용, 선택 아님)
    """
    for c in candidates:
        c["overfit"] = c["overfit_gap"] > overfit_threshold  # 진단용(선택 아님)
        c["dsr_pass"] = c.get("dsr", 0.0) >= dsr_threshold
    # IS 기반 DSR 내림차순, 동률은 IS Sharpe로 tie-break. OOS는 정렬 키에서 배제.
    return sorted(candidates, key=lambda c: (c.get("dsr", 0.0), c["is_sharpe"]), reverse=True)


def format_report(ranked: list[dict]) -> str:
    """후보 표 문자열. 정렬 기준=DSR(IS-only). OOS/Holdout은 진단용.

    각 후보 아래에 파라미터를 함께 출력 — 탐색이 어떤 피처를 실제로 채택했는지
    (예: entropy_weight가 0인지 아닌지) 리포트만으로 판독 가능해야 한다.
    """
    note = "정렬: DSR(IS 기반, 시도횟수 보정) 내림차순 — OOS/Holdout은 진단용(선택 아님)"
    header = f"{'#':<3}{'IS':>7}{'OOS':>7}{'Holdout':>9}{'MDD':>7}{'Trd':>5}{'DSR':>7}{'DSR?':>6}{'Overfit':>9}"
    lines = [note, header, "-" * len(header)]
    for i, c in enumerate(ranked, 1):
        flag = "⚠️" if c["overfit"] else "✅"
        dsr_flag = "✅" if c.get("dsr_pass") else "✗"
        lines.append(
            f"{i:<3}{c['is_sharpe']:>7.2f}{c['oos_sharpe']:>7.2f}"
            f"{c['holdout_sharpe']:>9.2f}{c['mdd']*100:>6.1f}%{c['trades']:>5}"
            f"{c.get('dsr', 0.0):>7.2f}{dsr_flag:>6}{flag:>9}"
        )
        params = c.get("params") or {}
        if params:
            body = ", ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(params.items())
            )
            lines.append(f"     └ {body}")
    return "\n".join(lines)
