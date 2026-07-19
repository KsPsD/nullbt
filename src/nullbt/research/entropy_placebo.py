"""엔트로피 필터 플라시보 대조 실험.

질문: 엔트로피 필터의 성과 개선(entropy_ab.py)이 '나쁜 거래를 선별'해서인가,
아니면 '아무 거래나 줄여서'인가? (음의 엣지 전략에서는 무엇이든 거래를 줄이면
성과가 0 쪽으로 개선되므로, 선별 능력은 무작위 제거와 비교해야만 판정 가능.)

설계:
- 엔트로피군: base + entropy_weight/window (기본 0.5/20 — A/B 최선 조합)
- 플라시보군: base 전략의 BUY 신호를 (seed, code, date) 해시 기반으로 확률 p 무작위
  억제 × N seeds. p는 엔트로피군과 거래수가 비슷해지도록 신호 억제율로 근사.
- 판정: 엔트로피군 성과가 플라시보 분포의 어느 백분위인지. 상위 꼬리(≥95%)면
  선별 능력 인정, 분포 중앙이면 기각.

실행: uv run python scripts/entropy_placebo.py [--seeds 20] [--suppress-p 0.86]
"""

import argparse
import hashlib
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nullbt.data import OHLCVStore, compute_trading_value, select_universe
from nullbt.engine import BacktestConfig, run_backtest
from nullbt.metrics import max_drawdown, sharpe_ratio
from nullbt.examples.adapter import make_breakout_signal_fn
from nullbt.validation import split_three_way

BASE_PARAMS = {
    "confidence_threshold": 30,
    "confidence_cutoff": 30.0,
    "stop_pct": 0.05,
    "target_pct": 0.12,
    "max_hold_days": 10,
    "volume_threshold": 1.5,
}


def make_placebo_fn(base_fn, suppress_p: float, seed: int):
    """BUY 신호를 (seed, code, date) 해시 기반 확률 suppress_p로 억제.

    전역 RNG 대신 해시를 써서 결정론적(재현 가능)이고 날짜/종목 간 독립.
    """

    def fn(code, history, params):
        action, conf = base_fn(code, history, params)
        if action in ("BUY", "STRONG_BUY"):
            key = f"{seed}:{code}:{history.index[-1]}"
            h = int(hashlib.md5(key.encode()).hexdigest(), 16) / float(16**32)
            if h < suppress_p:
                return ("HOLD", 0.0)
        return (action, conf)

    return fn


def measure(price_data, train, test, signal_fn, params, config):
    r_is = run_backtest(price_data, train, signal_fn, params, config)
    r_oos = run_backtest(price_data, test, signal_fn, params, config)
    return {
        "is_sharpe": sharpe_ratio(r_is.equity.pct_change().dropna()),
        "is_mdd": max_drawdown(r_is.equity),
        "is_trades": len(r_is.trades),
        "oos_sharpe": sharpe_ratio(r_oos.equity.pct_change().dropna()),
        "oos_mdd": max_drawdown(r_oos.equity),
        "oos_trades": len(r_oos.trades),
    }


def _fmt(tag, m, extra=""):
    return (
        f"{tag:>12} | {m['is_sharpe']:>7.2f} {m['is_mdd']:>6.1%} {m['is_trades']:>6} | "
        f"{m['oos_sharpe']:>7.2f} {m['oos_mdd']:>6.1%} {m['oos_trades']:>7}  {extra}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ohlcv")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--trade-start", default="2022-01-01")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="시작 시드(기존 런에 시드를 이어 붙일 때; 판정표는 이 런 분량만 집계)")
    ap.add_argument("--suppress-p", type=float, default=0.86,
                    help="플라시보 신호 억제율(엔트로피군 거래수에 근사하도록)")
    ap.add_argument("--entropy-weight", type=float, default=0.5)
    ap.add_argument("--entropy-window", type=int, default=20)
    args = ap.parse_args()

    store = OHLCVStore(Path(args.data))
    codes = sorted(p.stem for p in Path(args.data).glob("*.parquet"))
    values = {}
    for code in codes:
        df = store.load(code)
        if len(df.loc[: args.trade_start]) < 60:
            continue
        values[code] = compute_trading_value(df, as_of=args.trade_start)
    universe = select_universe(values, top_n=args.top)
    price_data = {c: store.load(c).loc[args.trade_start:] for c in universe}
    all_dates = sorted({d for df in price_data.values() for d in df.index})
    train, test, _ = split_three_way(all_dates)
    base_fn = make_breakout_signal_fn()
    config = BacktestConfig()

    print(f"universe={len(universe)}, train={len(train)}d, test={len(test)}d, "
          f"seeds={args.seeds}, suppress_p={args.suppress_p}")
    print(f"{'군':>12} | {'IS_shp':>7} {'IS_mdd':>6} {'IS_trd':>6} | "
          f"{'OOS_shp':>7} {'OOS_mdd':>6} {'OOS_trd':>7}")

    t0 = time.time()
    control = measure(price_data, train, test, base_fn, dict(BASE_PARAMS), config)
    print(_fmt("control", control, f"({time.time()-t0:.0f}s)"), flush=True)

    t0 = time.time()
    ent_params = dict(
        BASE_PARAMS, entropy_weight=args.entropy_weight, entropy_window=args.entropy_window
    )
    entropy = measure(price_data, train, test, base_fn, ent_params, config)
    print(_fmt("entropy", entropy, f"({time.time()-t0:.0f}s)"), flush=True)

    placebos = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        t0 = time.time()
        fn = make_placebo_fn(base_fn, args.suppress_p, seed)
        m = measure(price_data, train, test, fn, dict(BASE_PARAMS), config)
        placebos.append(m)
        print(_fmt(f"placebo#{seed}", m, f"({time.time()-t0:.0f}s)"), flush=True)

    print("\n=== 판정 (엔트로피 vs 플라시보 분포) ===")
    for metric, better_high in [
        ("oos_sharpe", True), ("is_sharpe", True), ("oos_mdd", False), ("is_mdd", False),
    ]:
        vals = [p[metric] for p in placebos]
        ent = entropy[metric]
        beat = sum(1 for v in vals if (ent > v) == better_high or ent == v)
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        pct = beat / len(vals) * 100
        print(f"{metric:>11}: entropy={ent:+.3f}  placebo={mean:+.3f}±{sd:.3f}  "
              f"→ 플라시보 {len(vals)}개 중 {beat}개보다 우수 ({pct:.0f}퍼센타일)")
    print("\n해석 기준: OOS Sharpe/MDD가 ≥95퍼센타일이면 선별 능력 인정,")
    print("50퍼센타일 부근이면 '거래 억제 효과일 뿐' → 피처 기각.")


if __name__ == "__main__":
    main()
