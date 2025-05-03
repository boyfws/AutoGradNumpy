import pytest
from src.dtypes.scalar.Float32 import Float32
import math

TEST_VALUES = [(10 ** 6, 32131), (21, 236), (8971, 42)]


def test_init() -> None:
    """
    Test-s initialization of dtypes.
    """
    test_class = Float32(1)


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_add(x_1, x_2) -> None:
    test_class = Float32(x_1)
    b = test_class + x_2
    assert isinstance(b, Float32)
    assert b.item() == x_1 + x_2


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_add2(x_1, x_2) -> None:
    test_class = Float32(x_1)
    test_class2 = Float32(x_2)
    b = test_class + test_class2
    assert isinstance(b, Float32)
    assert b.item() == x_1 + x_2


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_radd(x_1, x_2) -> None:
    test_class = Float32(x_1)
    b = x_2 + test_class
    assert isinstance(b, Float32)
    assert b.item() == x_1 + x_2


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_sub(x_1, x_2) -> None:
    test_class = Float32(x_1)
    b = test_class - x_2
    assert isinstance(b, Float32)
    assert b.item() == x_1 - x_2


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_sub2(x_1, x_2) -> None:
    test_class = Float32(x_1)
    test_class2 = Float32(x_2)
    b = test_class - test_class2
    assert isinstance(b, Float32)
    assert b.item() == x_1 - x_2


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_rsub(x_1, x_2) -> None:
    test_class = Float32(x_1)
    b = x_2 - test_class
    assert isinstance(b, Float32)
    assert b.item() == x_2 - x_1


@pytest.mark.parametrize(
    "x_1,x_2", TEST_VALUES
)
def test_mul(x_1, x_2) -> None:
    test_class = Float32(x_1)
    b = test_class * 2
    assert isinstance(b, Float32)
    assert b.item() == x_1 * 2


@pytest.mark.parametrize("x_1, x_2", [(10, 2), (100, 25)])
def test_rmul(x_1, x_2) -> None:
    """test reverse multiplication (e.g., 5 * Float32(3))."""
    a = Float32(x_2)
    result = x_1 * a
    assert isinstance(result, Float32)
    assert result.value == x_1 * x_2


@pytest.mark.parametrize(
    "x_1,x_2", [(1, 2), (100, 20), (10, 12)]
)
def test_truediv(x_1,x_2) -> None:
    test_class = Float32(x_1)
    b = test_class / x_2
    assert isinstance(b, Float32)
    assert b.item() == pytest.approx(x_1 / x_2)


@pytest.mark.parametrize("x_1, x_2", [(10, 2), (100, 25)])
def test_rtruediv(x_1, x_2) -> None:
    """test reverse division (e.g., 10 / Float32(2))."""
    a = Float32(x_2)
    result = x_1 / a  # Вызывает __rtruediv__
    assert isinstance(result, Float32)
    assert result.value == pytest.approx(x_1 / x_2)


def test_negation_operation() -> None:
    """test that the negation operation works correctly."""
    a = Float32(5.0)
    b = -a  # Calls __neg__
    assert isinstance(b, Float32)
    assert b.item() == -5.0


@pytest.mark.parametrize("base,exp,expected", [
    (2.0, 3.0, 8.0),  # 2^3 = 8
    (5.0, 0.0, 1.0),  # Any number to power of 0 is 1
    (4.0, 0.5, 2.0),  # Square root
    (2.0, -1.0, 0.5),  # Negative exponent
])
def test_pow_operation(base, exp, expected):
    """test basic power operation (a ** b)"""
    a = Float32(base)
    result = a ** exp
    assert isinstance(result, Float32)
    assert result.item() == pytest.approx(expected)


@pytest.mark.parametrize("base,exp,expected", [
    (2.0, 3.0, 8.0),  # 3^2 = 9 (reverse operation)
    (4, 0.5, 2.0),  # 4^0.5 = 2 (sqrt)
])
def test_rpow_operation(base, exp, expected):
    """test reverse power operation (base ** a)"""
    a = Float32(exp)
    result = base ** a  # Calls __rpow__
    assert isinstance(result, Float32)
    assert result.item() == pytest.approx(expected)


def test_pow_edge_cases():
    """test special cases for power operations"""
    # Identity operation (x^1 = x)
    assert (Float32(5.0) ** 1.0).item() == pytest.approx(5.0)

    # Mathematical convention (0^0 = 1)
    assert (Float32(0.0) ** 0.0).item() == pytest.approx(1.0)

