"""nullbt 데모 — 정직한 백테스트 리포트.

예쁜 수익곡선을 보여주는 데모가 아니다. 그 곡선을 얼마나 믿으면 안 되는지 보여주는 데모다.
번들된 한국 대형주 15종목(2019~2025) 위에서 돌아간다. 연구용이며 실매매 신호가 아니다.
"""

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from nullbt import (
    BacktestConfig,
    bootstrap_metrics,
    cagr,
    deflated_sharpe,
    max_drawdown,
    run_backtest,
    sharpe_ratio,
    split_three_way,
)
from nullbt.examples.adapter import make_breakout_signal_fn
from nullbt.signals import make_meanrev_signal_fn

DATA_DIR = Path(__file__).parent / "sample_data"

st.set_page_config(page_title="nullbt — 정직한 백테스트", page_icon="🕳️", layout="wide")


@st.cache_data
def load_data():
    names = json.loads((DATA_DIR / "names.json").read_text(encoding="utf-8"))
    price = {}
    for code in names:
        df = pd.read_parquet(DATA_DIR / f"{code}.parquet")
        df.index = pd.DatetimeIndex(df.index)
        price[code] = df
    return price, names


PRICE, NAMES = load_data()
ALL_DATES = sorted({d for df in PRICE.values() for d in df.index})

# ---- 헤더 ----
st.title("🕳️ nullbt")
st.markdown("#### 네 엣지는 증명 전까지 null이다")
st.caption(
    "한국 대형주 15종목(2019~2025) 위에서 전략을 돌려보되, **결과를 곧이곧대로 믿지 않는 법**까지 "
    "보여주는 데모입니다. 연구용이며 실매매 신호가 아닙니다. · "
    "[GitHub](https://github.com/KsPsD/nullbt) · [배경 글](https://valuebridge.tistory.com/5)"
)

# ---- 사이드바: 전략/파라미터 ----
with st.sidebar:
    st.header("전략 설정")
    strat = st.selectbox("전략", ["돌파 모멘텀", "평균회귀"])

    d0, d1 = st.select_slider(
        "기간",
        options=[d.date() for d in ALL_DATES],
        value=(ALL_DATES[0].date(), ALL_DATES[-1].date()),
    )

    st.subheader("파라미터")
    cutoff = st.slider("신호 확신 컷오프", 0, 90, 30, 5)
    stop = st.slider("손절 %", 1, 20, 5) / 100
    target = st.slider("익절 %", 2, 40, 12) / 100
    max_hold = st.slider("최대 보유일", 3, 40, 10)

    ent_w = 0.0
    if strat == "돌파 모멘텀":
        ent_w = st.slider(
            "엔트로피 필터 강도", 0.0, 0.5, 0.0, 0.05,
            help="방향 엔트로피가 높은(choppy) 구간의 확신을 감쇠. 리서치 아크에서 "
            "유일하게 플라시보를 통과한 후보 필터입니다(증명은 아님).",
        )

    st.subheader("정직성 점검용")
    n_trials = st.number_input(
        "이 결과를 얻기까지 시도한 조합 수", min_value=1, max_value=5000, value=50,
        help="파라미터를 몇 번 바꿔봤는지. DSR이 이 숫자만큼 성과를 의심합니다. "
        "슬라이더를 만지작거린 횟수를 정직하게 넣으세요.",
    )
    run = st.button("백테스트 실행", type="primary", use_container_width=True)


def build_params():
    p = {
        "confidence_threshold": cutoff,
        "confidence_cutoff": float(cutoff),
        "stop_pct": stop,
        "target_pct": target,
        "max_hold_days": int(max_hold),
        "volume_threshold": 1.5,
        "entropy_weight": ent_w,
    }
    return p


def signal_fn():
    return make_breakout_signal_fn() if strat == "돌파 모멘텀" else make_meanrev_signal_fn()


if not run:
    st.info("← 왼쪽에서 전략을 고르고 **백테스트 실행**을 눌러보세요.")
    st.stop()

# ---- 백테스트 ----
dates = [d for d in ALL_DATES if d0 <= d.date() <= d1]
if len(dates) < 60:
    st.error("기간이 너무 짧습니다. 최소 60거래일 이상 선택하세요.")
    st.stop()

params = build_params()
res = run_backtest(PRICE, dates, signal_fn(), params, BacktestConfig())
rets = res.equity.pct_change().dropna()
shp = sharpe_ratio(rets)
mdd = max_drawdown(res.equity)
ret_cagr = cagr(res.equity)
n_obs = len(rets)

# ---- 결과: 수익곡선 ----
st.subheader("📈 결과 (겉모습)")
eq = res.equity.copy()
eq.index = pd.DatetimeIndex(eq.index)
st.line_chart((eq / eq.iloc[0]).rename("자산배수"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe", f"{shp:.2f}")
c2.metric("CAGR", f"{ret_cagr * 100:.1f}%")
c3.metric("최대낙폭", f"{mdd * 100:.1f}%")
c4.metric("거래수", f"{len(res.trades)}")

st.divider()

# ---- 회의적 리포트 (nullbt의 본체) ----
st.subheader("🔍 회의적 리포트 — 이 결과를 믿어도 될까요?")

# 1) DSR
dsr = deflated_sharpe(shp / math.sqrt(252), int(n_trials), n_obs)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**1. Deflated Sharpe (DSR)**")
    st.metric(f"{int(n_trials)}번 시도를 감안한 유의확률", f"{dsr:.3f}")
    if dsr >= 0.95:
        st.success("탐색 다중성을 감안해도 0이 아닐 확률이 높습니다. (그래도 증명은 아님)")
    elif dsr >= 0.5:
        st.warning("애매합니다. 시도 수를 정직하게 넣었는지 다시 보세요.")
    else:
        st.error("이 시도 횟수로는 우연과 구분되지 않습니다.")

# 2) 부트스트랩 CI
with col_b:
    st.markdown("**2. 블록 부트스트랩 신뢰구간**")
    if n_obs >= 30:
        bs = bootstrap_metrics(rets, n_resamples=500, expected_block=20, seed=0)
        st.metric("Sharpe 점추정", f"{bs.sharpe.point:.2f}",
                  help="아래 구간이 이 값이 얼마나 흔들리는지 보여줍니다.")
        st.write(f"95% CI: **{bs.sharpe.low:.2f} ~ {bs.sharpe.high:.2f}**")
        if bs.sharpe.low <= 0 <= bs.sharpe.high:
            st.error("신뢰구간이 0을 포함합니다 — 수익이 우연일 여지가 큽니다.")
        else:
            st.info("구간이 0을 넘지 않습니다. 점추정보다 이 구간을 보고하세요.")
    else:
        st.write("표본이 부족해 CI를 생략합니다(≥30일 필요).")

# 3) IS / OOS 붕괴 점검
st.markdown("**3. In-Sample vs Out-of-Sample**")
try:
    is_d, oos_d, hold_d = split_three_way(dates)
    is_r = run_backtest(PRICE, is_d, signal_fn(), params, BacktestConfig()).equity.pct_change().dropna()
    oos_r = run_backtest(PRICE, oos_d, signal_fn(), params, BacktestConfig()).equity.pct_change().dropna()
    is_s, oos_s = sharpe_ratio(is_r), sharpe_ratio(oos_r)
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("IS Sharpe", f"{is_s:.2f}")
    cc2.metric("OOS Sharpe", f"{oos_s:.2f}")
    cc3.metric("과적합 격차 (IS−OOS)", f"{is_s - oos_s:.2f}")
    if is_s > 0 and oos_s < is_s * 0.3:
        st.error("OOS에서 성과가 대부분 사라졌습니다 — 과적합 신호.")
except Exception as e:  # noqa: BLE001
    st.write(f"(분할 점검 생략: {e})")

# 4) 정직성 체크리스트
st.markdown("**4. 구조적 편향 점검**")
st.markdown(
    "- ✅ **Lookahead 없음**: 신호는 당일 종가, 체결은 다음날 시가.\n"
    "- ⚠️ **생존편향 있음**: 이 15종목은 *지금까지 살아남은* 대형주입니다. "
    "상장폐지·부실 종목이 빠져 있어 성과가 **상방으로 부풀려집니다.**\n"
    f"- 📏 **표본 {n_obs}일** (약 {n_obs / 252:.1f}년, 시장 사이클 1~2개). 결론을 과신하기엔 짧습니다."
)

st.divider()
st.caption(
    "결론: 좋아 보이는 결과가 나왔다면, 그게 진짜인지 위 4가지로 의심해보세요. "
    "nullbt는 좋은 전략을 만들어주지 않습니다. 나쁜 전략을 빨리 걸러낼 뿐입니다."
)
