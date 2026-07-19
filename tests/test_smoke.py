def test_packages_import():
    import optuna  # noqa: F401

    import nullbt  # noqa: F401
    import nullbt.examples  # noqa: F401
    import nullbt.explore  # noqa: F401
    import nullbt.research  # noqa: F401


def test_public_api_surface():
    """최상위 재수출이 깨지지 않는지 — 사용자가 실제로 쓰는 진입점."""
    from nullbt import (  # noqa: F401
        BacktestConfig,
        OHLCVStore,
        deflated_sharpe,
        run_backtest,
        split_three_way,
        walk_forward_windows,
    )
