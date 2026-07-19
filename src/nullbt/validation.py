"""시간순 데이터 분할 (누수 차단). 셔플 금지."""


def split_three_way(
    dates: list, train_frac: float = 0.6, test_frac: float = 0.25
) -> tuple[list, list, list]:
    """시간순 정렬된 dates를 (train, test, holdout)로 분할. holdout은 미개봉용."""
    n = len(dates)
    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    train = dates[:n_train]
    test = dates[n_train : n_train + n_test]
    holdout = dates[n_train + n_test :]
    return train, test, holdout


def walk_forward_windows(in_sample_dates: list, n_folds: int) -> list[list]:
    """in-sample 구간을 연속 등분 n_folds개 윈도우로 분할."""
    if n_folds <= 1:
        return [list(in_sample_dates)]
    n = len(in_sample_dates)
    size = n // n_folds
    windows = []
    for i in range(n_folds):
        start = i * size
        end = (i + 1) * size if i < n_folds - 1 else n
        windows.append(in_sample_dates[start:end])
    return windows
