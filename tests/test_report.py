import math

from nullbt.metrics import deflated_sharpe
from nullbt.explore.report import format_report, rank_candidates


def test_rank_prefers_stable_over_overfit():
    cands = [
        {"params": {"id": "overfit"}, "is_sharpe": 3.0, "oos_sharpe": 0.2,
         "holdout_sharpe": 0.1, "mdd": 0.1, "trades": 40, "overfit_gap": 2.8, "dsr": 0.1},
        {"params": {"id": "stable"}, "is_sharpe": 1.8, "oos_sharpe": 1.6,
         "holdout_sharpe": 1.4, "mdd": 0.12, "trades": 38, "overfit_gap": 0.2, "dsr": 0.8},
    ]
    ranked = rank_candidates(cands, overfit_threshold=1.0)
    assert ranked[0]["params"]["id"] == "stable"
    assert ranked[0]["overfit"] is False
    # 큰 격차 후보는 오버핏 플래그
    overfit_row = next(r for r in ranked if r["params"]["id"] == "overfit")
    assert overfit_row["overfit"] is True


def test_dsr_decreases_with_more_trials():
    """DSR must strictly decrease as n_trials grows (trial-count deflation is working).

    Uses the de-annualized per-period Sharpe as measure_candidate now passes:
    annualized_sharpe / sqrt(252). With a fixed candidate and more trials, expected_max_sharpe
    grows so DSR falls.
    """
    per_period_sharpe = 1.5 / math.sqrt(252)  # ~0.094 — a moderately good strategy
    n_obs = 200
    dsr_few = deflated_sharpe(per_period_sharpe, n_trials=2, n_obs=n_obs)
    dsr_many = deflated_sharpe(per_period_sharpe, n_trials=500, n_obs=n_obs)
    assert dsr_many < dsr_few, (
        f"DSR should decrease as n_trials grows but got dsr_few={dsr_few:.4f} dsr_many={dsr_many:.4f}"
    )


def test_rank_by_dsr_not_oos():
    # OOS는 더 높지만 DSR 낮은 후보 vs OOS 낮지만 DSR 높은 후보.
    # 구(旧) 로직(oos - overfit_gap)이면 high_oos가 1위였음 → 신 로직은 DSR로 선택해야 함.
    cands = [
        {"params": {"id": "high_oos_low_dsr"}, "is_sharpe": 2.0, "oos_sharpe": 2.5,
         "holdout_sharpe": 0.1, "mdd": 0.1, "trades": 30, "overfit_gap": -0.5, "dsr": 0.30},
        {"params": {"id": "low_oos_high_dsr"}, "is_sharpe": 1.5, "oos_sharpe": 0.8,
         "holdout_sharpe": 0.7, "mdd": 0.1, "trades": 30, "overfit_gap": 0.7, "dsr": 0.90},
    ]
    ranked = rank_candidates(cands)
    assert ranked[0]["params"]["id"] == "low_oos_high_dsr"  # OOS가 아니라 DSR로 선택


def test_dsr_pass_flag_uses_threshold():
    cands = [
        {"params": {"id": "pass"}, "is_sharpe": 1.0, "oos_sharpe": 0.9, "holdout_sharpe": 0.8,
         "mdd": 0.1, "trades": 30, "overfit_gap": 0.1, "dsr": 0.97},
        {"params": {"id": "fail"}, "is_sharpe": 1.0, "oos_sharpe": 0.9, "holdout_sharpe": 0.8,
         "mdd": 0.1, "trades": 30, "overfit_gap": 0.1, "dsr": 0.50},
    ]
    by_id = {c["params"]["id"]: c for c in rank_candidates(cands, dsr_threshold=0.95)}
    assert by_id["pass"]["dsr_pass"] is True
    assert by_id["fail"]["dsr_pass"] is False


def test_format_report_contains_headers():
    ranked = rank_candidates([
        {"params": {"id": "x"}, "is_sharpe": 1.0, "oos_sharpe": 0.9,
         "holdout_sharpe": 0.8, "mdd": 0.1, "trades": 30, "overfit_gap": 0.1, "dsr": 0.72},
    ])
    out = format_report(ranked)
    assert "OOS" in out and "Holdout" in out
    assert "DSR" in out


def test_format_report_shows_candidate_params():
    """리포트만으로 피처 채택 여부(예: entropy_weight)를 판독할 수 있어야 한다."""
    ranked = rank_candidates([
        {"params": {"entropy_weight": 0.25, "stop_pct": 0.05}, "is_sharpe": 1.0,
         "oos_sharpe": 0.9, "holdout_sharpe": 0.8, "mdd": 0.1, "trades": 30,
         "overfit_gap": 0.1, "dsr": 0.72},
    ])
    out = format_report(ranked)
    assert "entropy_weight=0.25" in out
    assert "stop_pct=0.05" in out
