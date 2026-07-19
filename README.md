# nullbt

> **네 엣지는 증명 전까지 null이다.**
> Anti-overfitting backtesting engine for Korean equities.

한국 주식 전략 백테스트의 가장 큰 적은 시장이 아니라 **자기기만**이다.
nullbt는 전략을 빛나게 하는 도구가 아니라, **전략을 죽이려고 최선을 다하는** 도구다.
그 공격에서 살아남은 전략만 믿을 가치가 있다.

## 왜 만들었나

리테일 백테스트의 95%는 다음 중 하나로 스스로를 속인다:

| 자기기만 | nullbt의 방어 |
|---|---|
| Lookahead 편향 (미래 데이터 참조) | T종가 신호 → **T+1 시가 체결** 이벤트 루프 |
| 파라미터 과최적화 | **DSR**(Deflated Sharpe Ratio) — 탐색 횟수만큼 성과를 깎아서 평가 |
| 전체 기간에 피팅 | **IS / OOS / holdout 시간순 3분할** + walk-forward |
| 우연을 엣지로 착각 | **플라시보 테스트** — 랜덤 신호와 비교해 통계적으로 구분되는지 검증 |
| 비용 무시 | 수수료 + 슬리피지 + **한국 매도 증권거래세(0.18%)** 기본 반영 |

## 설치

```bash
pip install nullbt            # 코어 엔진
pip install "nullbt[data]"    # + FinanceDataReader (일봉 데이터 수집)
```

## 빠른 시작

```python
import pandas as pd
from nullbt import BacktestConfig, run_backtest, sharpe_ratio, deflated_sharpe

# 1) 종목별 일봉 DataFrame (open/high/low/close/volume, DatetimeIndex)
price_data: dict[str, pd.DataFrame] = load_your_data()

# 2) 신호 함수: (종목코드, 그 시점까지의 히스토리, 파라미터) → (액션, 확신도)
def my_signal(code: str, history: pd.DataFrame, params: dict) -> tuple[str, float]:
    ma20 = history["close"].tail(20).mean()
    if history["close"].iloc[-1] > ma20 * (1 + params["band"]):
        return ("BUY", 80.0)
    return ("HOLD", 0.0)

# 3) 실행 — 신호는 T 종가, 체결은 T+1 시가. 미래는 절대 안 보여준다.
dates = sorted({d for df in price_data.values() for d in df.index})
result = run_backtest(price_data, dates, my_signal, {"band": 0.02},
                      BacktestConfig(max_positions=10))

print(f"Sharpe: {sharpe_ratio(result.daily_returns):.2f}")

# 4) 진짜 질문: 이 Sharpe가 '탐색을 100번 해서 나온 것'이어도 유의한가?
print(f"DSR: {deflated_sharpe(result.daily_returns, n_trials=100):.3f}")
```

### 시간순 3분할 검증

```python
from nullbt import split_three_way, walk_forward_windows

is_dates, oos_dates, holdout_dates = split_three_way(dates)
# In-Sample에서 탐색 → OOS에서 1차 검증 → holdout은 마지막에 딱 한 번.
```

### 전략 탐색 (Optuna 역탐색 + DSR 리포트)

```python
from nullbt.explore import StrategySpec, run_search, format_report
# 파라미터 공간을 선언하면 IS에서 탐색하고,
# OOS 성과·DSR·overfit-gap을 묶어 리포트한다. 좋아 보이는 후보가 아니라
# '탐색 횟수를 감안해도 살아남는' 후보를 찾는 게 목적.
```

### 플라시보 테스트

```python
from nullbt.research.entropy_placebo import make_placebo_fn
# 네 신호의 일부를 같은 빈도의 랜덤 신호로 바꿔서 돌려본다.
# 진짜와 플라시보의 성과가 구분되지 않으면 — 그 엣지는 없는 것이다.
```

## 정직한 한계 (읽고 쓰세요)

- **주문 실행 코드는 없다.** 이건 리서치 도구다. 자동매매에 그대로 쓰지 마라.
- **생존편향이 완전히 제거되지 않았다.** FinanceDataReader의 상장 목록은 '현재' 상장 종목만 반환한다 — 상폐 종목이 빠진 유니버스는 성과를 부풀린다. point-in-time 유니버스는 로드맵에 있다.
- 일봉 기반이다. 장중 전략·호가 시뮬레이션은 범위 밖.
- 백테스트가 좋다고 돈을 버는 게 아니다. **nullbt는 나쁜 전략을 거르는 필터이지, 좋은 전략을 만들어주는 기계가 아니다.**

## 설계 원칙

1. **엔진은 미래를 모른다** — 신호 함수에는 그 시점까지의 데이터만 전달된다.
2. **모든 성과 지표는 탐색 비용을 안다** — 몇 번 시도해서 나온 결과인지가 지표에 반영된다.
3. **거짓 양성이 거짓 음성보다 비싸다** — 애매하면 죽인다.

## 로드맵

- [ ] point-in-time 유니버스 (상폐 종목 포함)
- [ ] 플라시보 테스트 1급 API 승격 (`nullbt.placebo`)
- [ ] 리포트 시각화 (HTML)
- [ ] 예제 노트북 + 전략 부검 사례집

## License

MIT
