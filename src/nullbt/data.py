"""OHLCV 적재·저장·조회 + 유니버스 선정. 네트워크(fetch)와 순수 로직 분리."""

from pathlib import Path

import pandas as pd

_COLS = ["open", "high", "low", "close", "volume"]


def compute_trading_value(df: pd.DataFrame, window: int = 20, as_of=None) -> float:
    """직전 window일 평균 거래대금(close*volume).

    as_of 지정 시 해당 시점까지의 데이터만 사용(point-in-time) — 백테스트 유니버스
    선정에 쓸 때 미래참조를 막는다. 미지정(None)이면 전체 기간의 마지막 window일을
    사용하는데, 이는 유니버스 선정에 쓰면 '말단 유동성' 미래참조가 됨(KNOWN_LIMITATIONS.md #6).
    """
    if df is None or df.empty:
        return 0.0
    series = df if as_of is None else df.loc[:as_of]
    tv = (series["close"] * series["volume"]).tail(window)
    return float(tv.mean()) if len(tv) else 0.0


def select_universe(code_to_value: dict[str, float], top_n: int) -> list[str]:
    """거래대금 상위 top_n 코드 (값 내림차순, 동률은 코드 오름차순)."""
    ordered = sorted(code_to_value.items(), key=lambda kv: (-kv[1], kv[0]))
    return [code for code, _ in ordered[:top_n]]


class OHLCVStore:
    """종목별 일봉을 parquet으로 로컬 저장/조회."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, code: str) -> Path:
        return self.root / f"{code}.parquet"

    def has(self, code: str) -> bool:
        return self._path(code).exists()

    def save(self, code: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(code))

    def load(self, code: str) -> pd.DataFrame:
        df = pd.read_parquet(self._path(code))
        # Normalize to datetime64[us] unit; do NOT impose freq — real market data has gaps.
        df.index = pd.DatetimeIndex(df.index.as_unit("us"))
        return df


def fetch_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    """FinanceDataReader로 일봉 조회 → 소문자 컬럼 정규화. (네트워크; 단위테스트 제외)"""
    import FinanceDataReader as fdr

    raw = fdr.DataReader(code, start, end)
    raw = raw.rename(columns=str.lower)
    df = raw[[c for c in _COLS if c in raw.columns]].copy()
    df.index.name = "date"
    return df
