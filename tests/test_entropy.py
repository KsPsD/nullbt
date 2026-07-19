from nullbt.entropy import normalized_entropy, return_entropy


def test_normalized_entropy_zero_when_all_equal():
    assert normalized_entropy([5, 5, 5, 5]) == 0.0


def test_normalized_entropy_max_when_uniform_across_bins():
    # 3구간에 1개씩 균등 배치 → H_norm == 1
    assert abs(normalized_entropy([0.0, 1.0, 2.0], bins=3) - 1.0) < 1e-9


def test_normalized_entropy_small_samples_and_nan():
    assert normalized_entropy([1.0]) == 0.0
    assert normalized_entropy([float("nan"), float("nan")]) == 0.0
    assert normalized_entropy([]) == 0.0


def test_normalized_entropy_in_unit_range():
    vals = [0.1, -0.2, 0.05, -0.15, 0.3, -0.05, 0.0, 0.22]
    h = normalized_entropy(vals, bins=4)
    assert 0.0 <= h <= 1.0


def test_return_entropy_zero_for_constant_returns():
    # 매 스텝 정확히 2배(수익률 1.0 고정, 2의 거듭제곱은 부동소수 나눗셈 정확) → 엔트로피 0
    prices = [float(2 ** i) for i in range(1, 21)]  # 매 스텝 상승만 → 방향 일관
    assert return_entropy(prices, window=20) == 0.0


def test_return_entropy_higher_for_choppy_than_trend():
    trend = [100 + i for i in range(30)]  # 완만 상승 → 수익률 좁게 분포
    choppy = [100.0]
    for i in range(29):
        choppy.append(choppy[-1] * (1.10 if i % 2 == 0 else 0.91))  # 큰 등락 반복
    assert return_entropy(choppy, window=20) > return_entropy(trend, window=20)
