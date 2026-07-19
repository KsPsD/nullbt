"""nullbt — 네 엣지는 증명 전까지 null이다.

한국 주식 전략을 위한 안티-과최적화 백테스트 엔진.
lookahead 없는 이벤트 루프, DSR(deflated Sharpe), IS/OOS/holdout 시간순 분리,
walk-forward, 플라시보 테스트를 기본 내장한다.
"""

from nullbt.bootstrap import BootstrapResult, MetricCI, bootstrap_metrics
from nullbt.data import (
    OHLCVStore,
    compute_trading_value,
    fetch_ohlcv,
    select_universe,
)
from nullbt.engine import (
    BacktestConfig,
    BacktestResult,
    SignalFn,
    Trade,
    run_backtest,
)
from nullbt.entropy import normalized_entropy, return_entropy
from nullbt.metrics import (
    cagr,
    deflated_sharpe,
    expected_max_sharpe,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)
from nullbt.rotation import RankFn, run_rotation_backtest
from nullbt.validation import split_three_way, walk_forward_windows

__version__ = "0.2.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BootstrapResult",
    "MetricCI",
    "OHLCVStore",
    "RankFn",
    "SignalFn",
    "Trade",
    "bootstrap_metrics",
    "cagr",
    "compute_trading_value",
    "deflated_sharpe",
    "expected_max_sharpe",
    "fetch_ohlcv",
    "max_drawdown",
    "normalized_entropy",
    "return_entropy",
    "run_backtest",
    "run_rotation_backtest",
    "select_universe",
    "sharpe_ratio",
    "split_three_way",
    "walk_forward_windows",
    "win_rate",
]
