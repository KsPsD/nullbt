import pandas as pd

from nullbt.data import OHLCVStore, compute_trading_value, select_universe


def _df(closes, vols):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": vols},
        index=idx,
    )


def test_compute_trading_value_mean_of_close_times_volume():
    df = _df([10, 20], [100, 200])  # 10*100=1000, 20*200=4000 -> mean 2500
    assert compute_trading_value(df, window=2) == 2500.0


def test_select_universe_top_n_by_value():
    vals = {"A": 100.0, "B": 300.0, "C": 200.0}
    assert select_universe(vals, top_n=2) == ["B", "C"]


def test_compute_trading_value_as_of_is_point_in_time():
    # 3거래일: 2024-01-01/02/03, 거래대금 1000/2000/3000
    df = _df([10, 20, 30], [100, 100, 100])
    # as_of=01-02 → 앞 2일만 사용(미래참조 제거): mean(1000, 2000) = 1500
    assert compute_trading_value(df, window=20, as_of="2024-01-02") == 1500.0
    # as_of가 전체 데이터보다 앞서면 표본 없음 → 0.0 (NaN 아님)
    assert compute_trading_value(df, as_of="2023-01-01") == 0.0
    # as_of 미지정이면 전체 마지막 window 사용(기존 동작 유지)
    assert compute_trading_value(df, window=3) == 2000.0


def test_store_roundtrip(tmp_path):
    store = OHLCVStore(tmp_path)
    df = _df([10, 11, 12], [100, 100, 100])
    assert store.has("A") is False
    store.save("A", df)
    assert store.has("A") is True
    loaded = store.load("A")
    # check_freq=False: parquet does not preserve DatetimeIndex.freq metadata.
    # Values, columns, and index dates must match exactly.
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_store_roundtrip_with_gapped_index(tmp_path):
    """load() must not raise when the index has gaps (e.g. trading days skip weekends)."""
    store = OHLCVStore(tmp_path)
    # Simulate 5 trading days: Mon-Fri of one week, skipping the following weekend.
    # This index has freq=None because Sat/Sun are missing — freq="D" would raise.
    dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                            "2024-01-15"])  # skip Jan 13/14 (weekend)
    df = pd.DataFrame(
        {"open": [100] * 6, "high": [110] * 6, "low": [90] * 6,
         "close": [105] * 6, "volume": [1000] * 6},
        index=pd.DatetimeIndex(dates),
    )
    df.index.name = "date"
    store.save("B", df)
    loaded = store.load("B")  # must not raise ValueError
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)
