"""전략탐색기 end-to-end 엔트리. 기존 main.py와 독립(additive)."""

import math
import statistics

from nullbt.engine import BacktestConfig
from nullbt.examples.adapter import make_breakout_signal_fn
from nullbt.validation import split_three_way, walk_forward_windows
from nullbt.explore.report import format_report, measure_candidate, rank_candidates
from nullbt.explore.search import run_search
from nullbt.explore.spec import StrategySpec

_ENTROPY_SPACE = {
    # 정보이론 레짐 필터(플라시보 검증됨): 0이면 무효과 → 탐색기가 채택 여부 판정.
    "entropy_weight": (0.0, 0.5),
    "entropy_window": (10, 40),
}

_DEFAULT_SPEC = StrategySpec(
    constraints={"max_mdd": 0.15, "min_trades": 10},
    search_space={
        "confidence_threshold": (20, 55),
        "confidence_cutoff": (20.0, 50.0),
        "stop_pct": (0.03, 0.10),
        "target_pct": (0.05, 0.25),
        "max_hold_days": (3, 20),
        "volume_threshold": (1.2, 3.0),
        **_ENTROPY_SPACE,
    },
)

_MEANREV_SPEC = StrategySpec(
    constraints={"max_mdd": 0.15, "min_trades": 10},
    search_space={
        "mr_window": (5, 20),
        "z_entry": (1.0, 3.0),
        "use_trend": (0, 1),
        "trend_window": (50, 150),
        "confidence_cutoff": (20.0, 80.0),
        "stop_pct": (0.02, 0.08),
        "target_pct": (0.03, 0.15),
        "max_hold_days": (2, 10),
        **_ENTROPY_SPACE,
    },
)

_DIPBUY_SPEC = StrategySpec(
    constraints={"max_mdd": 0.15, "min_trades": 10},
    search_space={
        "drop_pct": (0.03, 0.10),
        "vol_mult": (1.0, 3.0),
        "confidence_cutoff": (20.0, 80.0),
        "stop_pct": (0.02, 0.08),
        "target_pct": (0.03, 0.15),
        "max_hold_days": (2, 10),
        **_ENTROPY_SPACE,
    },
)


_ROT_BASE_SPACE = {
    "mom_window": (40, 250),
    "skip_days": (0, 20),
    "top_k": (5, 20),
    "rebalance_days": (5, 21),
}

# 로테이션은 장기 보유 롱온리 포트폴리오라 시장 수준 MDD를 허용(0.30).
_ROT_PLAIN_SPEC = StrategySpec(
    constraints={"max_mdd": 0.30, "min_trades": 20},
    search_space=dict(_ROT_BASE_SPACE),
)

_ROT_ENT_SPEC = StrategySpec(
    constraints={"max_mdd": 0.30, "min_trades": 20},
    search_space={
        **_ROT_BASE_SPACE,
        # 엔트로피 '사전선별'(감쇠 아님): 어수선한 종목을 순위에서 제외.
        # rot_plain과의 차이가 이 두 파라미터뿐 → 패밀리 수준 ablation.
        "ent_max": (0.55, 0.95),
        "entropy_window": (10, 40),
    },
)


def _family(name: str):
    """전략 패밀리 → (signal_fn 팩토리, 탐색 스펙). 새 패밀리는 여기 등록.
    rot_* 패밀리는 rotation 엔진 경로(main_local에서 분기)."""
    from nullbt.signals import (
        make_dipbuy_signal_fn,
        make_meanrev_signal_fn,
        make_momentum_rank_fn,
    )

    return {
        "breakout": (make_breakout_signal_fn, _DEFAULT_SPEC),
        "meanrev": (make_meanrev_signal_fn, _MEANREV_SPEC),
        "dipbuy": (make_dipbuy_signal_fn, _DIPBUY_SPEC),
        "rot_plain": (make_momentum_rank_fn, _ROT_PLAIN_SPEC),
        "rot_ent": (make_momentum_rank_fn, _ROT_ENT_SPEC),
    }[name]


def run_exploration(
    price_data: dict,
    n_trials: int = 30,
    n_folds: int = 3,
    top_k: int = 5,
    config: BacktestConfig | None = None,
    spec: StrategySpec = _DEFAULT_SPEC,
    signal_fn=None,
    dsr_trials: int | None = None,
) -> str:
    """dsr_trials: DSR 시도횟수 보정에 쓸 값. 여러 패밀리를 같은 데이터에 탐색하는
    캠페인에서는 이 스터디의 n_trials가 아니라 캠페인 전체 trial 수를 넣어야
    다중비교가 정직하게 보정된다."""
    config = config or BacktestConfig()
    signal_fn = signal_fn or make_breakout_signal_fn()

    # 공통 거래일 축 = 가장 긴 종목의 인덱스
    all_dates = sorted({d for df in price_data.values() for d in df.index})
    train, test, holdout = split_three_way(all_dates)
    windows = walk_forward_windows(train, n_folds)

    # 1) in-sample 한정 탐색
    study = run_search(spec, price_data, windows, signal_fn, config, n_trials)

    # 1.5) 교차-trial Sharpe 분산(per-period) 산출 — DSR의 SR0 허들 보정용.
    #      하드코딩 1.0 대신 실제 탐색이 훑은 Sharpe 분산을 사용(Bailey & López de Prado).
    completed = [t for t in study.trials if t.value is not None]
    trial_sharpes = [t.user_attrs.get("is_sharpe", 0.0) for t in completed]
    if len(trial_sharpes) >= 2:
        # trial Sharpe는 연율화값 → per-period로 환산(/sqrt(252)) 후 표본표준편차
        sharpe_std = statistics.stdev(trial_sharpes) / math.sqrt(252)
        if sharpe_std <= 0:
            sharpe_std = 1.0
    else:
        sharpe_std = 1.0

    # 2) 상위 top_k 후보를 IS/OOS/holdout 측정 (top_k 선별은 IS 탐색점수 기준 — OOS 미개입)
    trials = sorted(study.trials, key=lambda t: (t.value if t.value is not None else -1e9), reverse=True)
    candidates = []
    for t in trials[:top_k]:
        cand = measure_candidate(
            t.params, windows, test, holdout, price_data, signal_fn, config,
            n_trials=dsr_trials or n_trials, sharpe_std=sharpe_std,
        )
        candidates.append(cand)

    # 3) 오버핏 격차 순 정렬 + 리포트
    ranked = rank_candidates(candidates)
    return format_report(ranked)


def main_local(
    data_dir: str = "data/ohlcv",
    trade_start: str = "2022-01-01",
    top_n: int = 50,
    n_trials: int = 50,
    family: str = "breakout",
    dsr_trials: int | None = None,
) -> None:  # pragma: no cover - 파일시스템 CLI
    """로컬 parquet 저장소만으로 탐색 실행(네트워크 불필요).

    데이터는 사전에 수집해 둔다(예: `python -m src.collectors.kis_daily_history`).
    유동성 스크리닝은 main()과 동일하게 trade_start 이전 데이터만 사용(point-in-time).
    """
    from pathlib import Path

    from nullbt.data import OHLCVStore, compute_trading_value, select_universe

    root = Path(data_dir)
    codes = sorted(p.stem for p in root.glob("*.parquet"))
    if not codes:
        raise SystemExit(f"{data_dir} 에 parquet 없음 — 수집기를 먼저 실행할 것")
    store = OHLCVStore(root)

    values = {}
    for code in codes:
        df = store.load(code)
        # trade_start 이전 이력이 충분한 종목만(신규상장 제외) 유동성 측정
        pre = df.loc[:trade_start]
        if len(pre) < 60:
            continue
        values[code] = compute_trading_value(df, as_of=trade_start)
    universe = select_universe(values, top_n=top_n)
    price_data = {c: store.load(c).loc[trade_start:] for c in universe}
    n_days = max(len(df) for df in price_data.values())
    fn_factory, spec = _family(family)
    print(f"유니버스 {len(universe)}종목 (후보 {len(values)}), 거래일 ~{n_days}일, "
          f"family={family}, trials={n_trials}, dsr_trials={dsr_trials or n_trials}")
    if family.startswith("rot_"):
        from nullbt.explore.rotation import run_rotation_exploration

        print(run_rotation_exploration(price_data, rank_fn=fn_factory(), spec=spec,
                                       n_trials=n_trials, dsr_trials=dsr_trials))
    else:
        print(run_exploration(price_data, n_trials=n_trials, spec=spec,
                              signal_fn=fn_factory(), dsr_trials=dsr_trials))


def main() -> None:  # pragma: no cover - 네트워크 CLI
    from datetime import date

    import FinanceDataReader as fdr

    from nullbt.data import (
        OHLCVStore,
        compute_trading_value,
        fetch_ohlcv,
        select_universe,
    )

    # ⚠️ 잔존 생존편향: StockListing은 '현재' 상장 종목만 반환(상폐/합병 종목 제외).
    #    완전한 point-in-time 유니버스는 과거 구성종목(상폐 포함) 소스가 필요함 → 미해결.
    #    아래는 최소 개선: 유동성 스크리닝을 '거래 시작 시점'까지의 데이터로만 수행해
    #    말단 유동성 미래참조를 제거(KNOWN_LIMITATIONS.md #6).
    listing = fdr.StockListing("KOSPI")
    codes = listing["Code"].tolist()[:100]
    store = OHLCVStore("data/ohlcv")

    screen_start = "2021-07-01"  # 유동성 스크리닝용 사전 기간
    trade_start = "2022-01-01"   # 실제 백테스트 거래 시작 시점
    end = date.today().isoformat()

    values = {}
    for code in codes:
        if not store.has(code):
            try:
                store.save(code, fetch_ohlcv(code, screen_start, end))
            except Exception:  # noqa: BLE001
                continue
        if store.has(code):
            # 거래 시작 시점까지의 데이터로만 유동성 측정 → 미래참조 제거
            values[code] = compute_trading_value(store.load(code), as_of=trade_start)
    universe = select_universe(values, top_n=50)
    # 거래는 trade_start 이후 구간만 사용(스크리닝 사전기간은 백테스트에서 제외)
    price_data = {c: store.load(c).loc[trade_start:] for c in universe}
    print(run_exploration(price_data, n_trials=50))


if __name__ == "__main__":
    import sys

    if "--local" in sys.argv:
        kwargs = {}
        for arg in sys.argv[1:]:
            if arg.startswith("--trials="):
                kwargs["n_trials"] = int(arg.split("=", 1)[1])
            elif arg.startswith("--top="):
                kwargs["top_n"] = int(arg.split("=", 1)[1])
            elif arg.startswith("--family="):
                kwargs["family"] = arg.split("=", 1)[1]
            elif arg.startswith("--dsr-trials="):
                kwargs["dsr_trials"] = int(arg.split("=", 1)[1])
            elif arg.startswith("--trade-start="):
                kwargs["trade_start"] = arg.split("=", 1)[1]
        main_local(**kwargs)
    else:
        main()
