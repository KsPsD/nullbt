from nullbt.validation import split_three_way, walk_forward_windows


def test_split_three_way_partitions_without_overlap():
    dates = list(range(100))
    train, test, holdout = split_three_way(dates, 0.6, 0.25)
    assert len(train) == 60
    assert len(test) == 25
    assert len(holdout) == 15
    # 시간순 + 서로소 + 합집합 == 입력
    assert train + test + holdout == dates
    assert max(train) < min(test) < min(holdout)


def test_walk_forward_windows_contiguous_equal():
    dates = list(range(60))
    windows = walk_forward_windows(dates, n_folds=3)
    assert len(windows) == 3
    assert all(len(w) == 20 for w in windows)
    assert windows[0] + windows[1] + windows[2] == dates


def test_walk_forward_single_fold_returns_all():
    dates = list(range(10))
    assert walk_forward_windows(dates, 1) == [dates]
