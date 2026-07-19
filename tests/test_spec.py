from nullbt.explore.spec import StrategySpec, penalty, sample_params


def test_penalty_zero_when_constraints_met():
    m = {"mdd": 0.10, "trades": 50, "win_rate": 0.5}
    c = {"max_mdd": 0.15, "min_trades": 30}
    assert penalty(m, c) == 0.0


def test_penalty_positive_when_mdd_exceeded():
    m = {"mdd": 0.30, "trades": 50, "win_rate": 0.5}
    c = {"max_mdd": 0.15, "min_trades": 30}
    assert penalty(m, c) > 0.0


def test_penalty_large_when_too_few_trades():
    m = {"mdd": 0.05, "trades": 5, "win_rate": 0.5}
    c = {"max_mdd": 0.15, "min_trades": 30}
    few = penalty(m, c)
    m2 = {"mdd": 0.30, "trades": 50, "win_rate": 0.5}
    mdd_only = penalty(m2, c)
    assert few > mdd_only  # 거래부족 페널티가 더 큼


class _FakeTrial:
    def suggest_int(self, name, lo, hi):
        return lo
    def suggest_float(self, name, lo, hi):
        return lo


def test_sample_params_types():
    space = {"max_hold_days": (3, 20), "stop_pct": (0.03, 0.10)}
    p = sample_params(_FakeTrial(), space)
    assert p["max_hold_days"] == 3
    assert p["stop_pct"] == 0.03


def test_spec_defaults_present():
    spec = StrategySpec(constraints={"max_mdd": 0.15, "min_trades": 30},
                        search_space={"confidence_cutoff": (20.0, 50.0)})
    assert spec.objective == "sharpe"
