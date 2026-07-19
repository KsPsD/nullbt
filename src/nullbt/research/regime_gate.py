"""포트폴리오 레벨 엔트로피 레짐 게이트 실험.

배경: 플라시보 대조(entropy_placebo.py)에서 종목 단위 방향 엔트로피 필터의
OOS 선별 능력이 인정됨(98퍼센타일). 이번엔 같은 가설을 더 단순한 형태로 승격:
  "시장 전체가 어수선한 날엔 신규 진입 자체를 끈다" (on/off 게이트, 파라미터 1개)

사전 등록(선택편향 방지):
- window=20 (플라시보 검증에서 이월, 재탐색 없음)
- 주 검정 임계값 = train 구간 시장 엔트로피의 q70 분위수 (아래 PRIMARY_Q)
- 판정 = 회전(circular shift) 플라시보 20개 대비 OOS Sharpe/MDD ≥95퍼센타일
  (회전 플라시보: 시장 엔트로피 시계열을 통째로 k일 회전 → 자기상관 구조는
   보존하면서 실제 장세와의 정렬만 파괴. 무작위 날짜 게이트보다 엄격한 대조)
- q50~q90 그리드는 용량-반응 진단용(선택 아님)

실행: uv run python scripts/regime_gate.py [--top 50] [--rotations 20]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

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
WINDOW = 20        # 사전 등록: 플라시보 검증에서 이월
PRIMARY_Q = 0.70   # 사전 등록: 주 검정 임계 분위수
DIAG_QS = [0.50, 0.60, 0.70, 0.80, 0.90]  # 용량-반응 진단(선택 아님)


def rolling_direction_entropy(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """종목 1개의 rolling 방향 엔트로피(∈[0,1]) — features.entropy.return_entropy의
    벡터화 등가물(테스트로 일치 검증). 인과적(rolling, 미래 미참조)."""
    r = close.pct_change()
    valid = r.notna().astype(float)
    up = ((r > 0) & r.notna()).astype(float)
    dn = ((r < 0) & r.notna()).astype(float)
    n = valid.rolling(window).sum()
    cu, cd = up.rolling(window).sum(), dn.rolling(window).sum()
    cf = n - cu - cd
    h = pd.Series(0.0, index=close.index)
    for c in (cu, cd, cf):
        p = (c / n).where(n > 0)
        term = -(p * p.map(lambda x: math.log(x) if x and x > 0 else 0.0)).fillna(0.0)
        h = h + term
    h = (h / math.log(3)).where(n >= 2)
    return h


def market_entropy(price_data: dict, window: int = WINDOW) -> pd.Series:
    """유니버스 평균 방향 엔트로피(일별). 종목별 rolling 엔트로피의 횡단면 평균."""
    cols = {c: rolling_direction_entropy(df["close"], window) for c, df in price_data.items()}
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True)


def gate_dates_from(me: pd.Series, train_dates: list, q: float) -> tuple[set, float]:
    """train 구간 분위수로 임계값을 고정(PIT)한 뒤 전 기간에 적용."""
    thr = float(me.loc[me.index.isin(train_dates)].quantile(q))
    return set(me.index[me > thr]), thr


def make_gated_fn(base_fn, gated: set):
    def fn(code, history, params):
        action, conf = base_fn(code, history, params)
        if action in ("BUY", "STRONG_BUY") and history.index[-1] in gated:
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
        f"{tag:>14} | {m['is_sharpe']:>7.2f} {m['is_mdd']:>6.1%} {m['is_trades']:>6} | "
        f"{m['oos_sharpe']:>7.2f} {m['oos_mdd']:>6.1%} {m['oos_trades']:>7}  {extra}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ohlcv")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--trade-start", default="2022-01-01")
    ap.add_argument("--rotations", type=int, default=20)
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

    me = market_entropy(price_data)
    print(f"universe={len(universe)}, train={len(train)}d, test={len(test)}d, "
          f"window={WINDOW}, primary_q={PRIMARY_Q}, rotations={args.rotations}")
    print(f"시장 엔트로피: median={me.median():.3f}, q70={me.quantile(0.7):.3f}, "
          f"결측 {me.isna().sum()}일")
    print(f"{'군':>14} | {'IS_shp':>7} {'IS_mdd':>6} {'IS_trd':>6} | "
          f"{'OOS_shp':>7} {'OOS_mdd':>6} {'OOS_trd':>7}")

    t0 = time.time()
    control = measure(price_data, train, test, base_fn, dict(BASE_PARAMS), config)
    print(_fmt("control", control, f"({time.time()-t0:.0f}s)"), flush=True)

    # ── 용량-반응 진단 그리드 (선택 아님) ──
    primary = None
    for q in DIAG_QS:
        gated, thr = gate_dates_from(me, train, q)
        t0 = time.time()
        m = measure(price_data, train, test, make_gated_fn(base_fn, gated),
                    dict(BASE_PARAMS), config)
        gated_frac = len([d for d in test if d in gated]) / len(test)
        tag = f"gate@q{int(q*100)}"
        note = f"(thr={thr:.3f}, OOS게이트 {gated_frac:.0%}, {time.time()-t0:.0f}s)"
        if abs(q - PRIMARY_Q) < 1e-9:
            primary = m
            tag += "*"
        print(_fmt(tag, m, note), flush=True)

    # ── 회전 플라시보 (주 검정: q70 게이트 vs 정렬 파괴 게이트) ──
    print(f"\n회전 플라시보 ×{args.rotations} (엔트로피 시계열 k일 원형 회전, "
          f"임계값은 회전본의 train 분위수로 재산출 → 게이트 비율 동등)")
    n = len(me)
    offsets = [int(n * (i + 1) / (args.rotations + 1)) for i in range(args.rotations)]
    placebos = []
    for i, k in enumerate(offsets):
        rot = pd.Series(list(me.iloc[k:]) + list(me.iloc[:k]), index=me.index)
        gated, _ = gate_dates_from(rot, train, PRIMARY_Q)
        t0 = time.time()
        m = measure(price_data, train, test, make_gated_fn(base_fn, gated),
                    dict(BASE_PARAMS), config)
        placebos.append(m)
        print(_fmt(f"rot#{i}(k={k})", m, f"({time.time()-t0:.0f}s)"), flush=True)

    print("\n=== 판정 (q70 게이트 vs 회전 플라시보 분포) ===")
    import statistics
    for metric, better_high in [
        ("oos_sharpe", True), ("is_sharpe", True), ("oos_mdd", False), ("is_mdd", False),
    ]:
        vals = [p[metric] for p in placebos]
        real = primary[metric]
        beat = sum(1 for v in vals if (real > v) == better_high or real == v)
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{metric:>11}: gate={real:+.3f}  rot={mean:+.3f}±{sd:.3f}  "
              f"→ {len(vals)}개 중 {beat}개보다 우수 ({beat/len(vals)*100:.0f}퍼센타일)")
    print("\n사전 기준: OOS Sharpe/MDD ≥95퍼센타일이면 게이트 승격 인정.")


if __name__ == "__main__":
    main()
