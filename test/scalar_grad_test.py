import pytest
import math

from add_src_to_path import append_src
append_src()

from src import Float32

# test values (integers for precision)
TEST_VALUES = [(10, 3), (100, 20), (15, 5)]
TEST_VALUES_DIV = [(10, 2), (100, 25), (15, 3)]  # Avoid division by zero


def test_init() -> None:
    """test scalar initialization."""
    x = Float32(5, requires_grad=True)
    assert x.item() == 5
    assert x.requires_grad is True
    assert x.grad is None


@pytest.mark.parametrize("x_1, x_2", TEST_VALUES)
def test_add_grad(x_1, x_2) -> None:
    """test gradient computation for addition operation."""
    a = Float32(x_1, requires_grad=True)
    b = Float32(x_2, requires_grad=True)
    c = a + b
    c.backward()

    assert c.grad is None  # Gradient of the graph's root is not stored after backward()
    assert a.grad == 1  # dc/da = 1
    assert b.grad == 1  # dc/db = 1


@pytest.mark.parametrize("x_1, x_2", TEST_VALUES)
def test_sub_grad(x_1, x_2) -> None:
    """test gradient computation for subtraction operation."""
    a = Float32(x_1, requires_grad=True)
    b = Float32(x_2, requires_grad=True)
    c = a - b
    c.backward()

    assert a.grad == 1  # dc/da = 1
    assert b.grad == -1  # dc/db = -1


@pytest.mark.parametrize("x_1, x_2", TEST_VALUES)
def test_mul_grad(x_1, x_2) -> None:
    """test gradient computation for multiplication operation."""
    a = Float32(x_1, requires_grad=True)
    b = Float32(x_2, requires_grad=True)
    c = a * b
    c.backward()

    assert a.grad == x_2  # dc/da = b
    assert b.grad == x_1  # dc/db = a


@pytest.mark.parametrize("x_1, x_2", TEST_VALUES_DIV)
def test_div_grad(x_1, x_2) -> None:
    """test gradient computation for division operation."""
    a = Float32(x_1, requires_grad=True)
    b = Float32(x_2, requires_grad=True)
    c = a / b
    c.backward()

    assert a.grad == pytest.approx(1 / x_2)  # dc/da = 1/b
    assert b.grad == pytest.approx(-x_1 / (x_2 ** 2))  # dc/db = -a/(b^2)


def test_chain_rule() -> None:
    """test chain rule (combined operations)."""
    a = Float32(2, requires_grad=True)
    b = Float32(3, requires_grad=True)
    c = a * b  # c = 2 * 3 = 6
    d = c + 1  # d = 6 + 1 = 7
    d.backward()

    assert d.item() == 7
    assert a.grad == 3  # dd/da = b = 3
    assert b.grad == 2  # dd/db = a = 2


def test_no_grad() -> None:
    """test disabled gradients (requires_grad=False)."""
    a = Float32(5, requires_grad=False)
    b = Float32(10, requires_grad=True)
    c = a * b
    c.backward()

    assert a.grad is None  # requires_grad=False
    assert b.grad == 5  # dc/db = a = 5


def test_grad_accumulation() -> None:
    """test gradient accumulation across multiple backward passes."""
    a = Float32(4, requires_grad=True)
    b = Float32(2, requires_grad=True)

    # First pass
    c = a * b
    c.backward()
    assert a.grad == 2
    assert b.grad == 4

    # Second pass (gradients should accumulate)
    d = a + b
    d.backward()
    assert a.grad == 3  # 2 (previous) + 1 (new)
    assert b.grad == 5  # 4 + 1


def test_negation_gradient() -> None:
    """test gradient computation through negation operation."""
    a = Float32(3.0, requires_grad=True)
    b = -a  # b = -3.0
    b.backward()
    assert b.item() == -3.0
    assert a.grad == -1.0  # db/da = -1

def test_negation_chain_rule() -> None:
    """test chain rule through multiple operations including negation."""
    a = Float32(2.0, requires_grad=True)
    b = Float32(4.0, requires_grad=True)
    c = -a    # c = -2.0
    d = c * b  # d = -2.0 * 4.0 = -8.0
    d.backward()
    assert d.item() == -8.0
    assert a.grad == -4.0  # dd/da = -b = -4.0
    assert b.grad == -2.0  # dd/db = -a = -2.0

def test_double_negation() -> None:
    """test that double negation works and computes gradients correctly."""
    a = Float32(1.5, requires_grad=True)
    b = -(-a)  # b = 1.5
    b.backward()
    assert b.item() == pytest.approx(1.5)
    assert a.grad == 1.0  # db/da = 1 (derivative of -(-x) is 1)

def test_negation_with_constants() -> None:
    """test negation when combined with constant values."""
    a = Float32(3.0, requires_grad=True)
    b = 2.0 - (-a)  # b = 2.0 - (-3.0) = 5.0
    b.backward()
    assert b.item() == 5.0
    assert a.grad == 1.0  # db/da = 1


# test data - (left_operand, right_operand, expected_result)
RADD_TEST_VALUES = [(5, 3, 8), (10.5, 2.5, 13.0)]
RSUB_TEST_VALUES = [(10, 4, 6), (8.0, 3.0, 5.0)]
RMUL_TEST_VALUES = [(4, 3, 12), (2.5, 4.0, 10.0)]
RTRUEDIV_TEST_VALUES = [(10, 2, 5.0), (9.0, 3.0, 3.0)]


def test_radd_operation_and_gradient():
    """test __radd__ operation and gradient computation"""
    for left, right, expected in RADD_TEST_VALUES:
        # test operation
        a = Float32(right, requires_grad=True)
        result = left + a  # Calls __radd__

        assert isinstance(result, Float32)
        assert result.item() == pytest.approx(expected)

        # test gradient
        result.backward()
        assert a.grad == 1.0  # d(result)/da = 1


def test_rsub_operation_and_gradient():
    """test __rsub__ operation and gradient computation"""
    for left, right, expected in RSUB_TEST_VALUES:
        # test operation
        a = Float32(right, requires_grad=True)
        result = left - a  # Calls __rsub__

        assert isinstance(result, Float32)
        assert result.item() == pytest.approx(expected)

        # test gradient
        result.backward()
        assert a.grad == -1.0  # d(result)/da = -1


def test_rmul_operation_and_gradient():
    """test __rmul__ operation and gradient computation"""
    for left, right, expected in RMUL_TEST_VALUES:
        # test operation
        a = Float32(right, requires_grad=True)
        result = left * a  # Calls __rmul__

        assert isinstance(result, Float32)
        assert result.item() == pytest.approx(expected)

        # test gradient
        result.backward()
        assert a.grad == pytest.approx(left)  # d(result)/da = left_operand


def test_rtruediv_operation_and_gradient():
    """test __rtruediv__ operation and gradient computation"""
    for left, right, expected in RTRUEDIV_TEST_VALUES:
        # test operation
        a = Float32(right, requires_grad=True)
        result = left / a  # Calls __rtruediv__

        assert isinstance(result, Float32)
        assert result.item() == pytest.approx(expected)

        # test gradient
        result.backward()
        expected_grad = -left / (right ** 2)
        assert a.grad == pytest.approx(expected_grad)  # d(result)/da = -left/a²


def test_mixed_reverse_operations():
    """test combination of reverse operations with gradients"""
    a = Float32(2.0, requires_grad=True)
    b = Float32(3.0, requires_grad=True)

    # Complex expression: (5 + a) * (4 - b) / 2
    result = (5 + a) * (4 - b) / 2.0

    assert isinstance(result, Float32)
    assert result.item() == pytest.approx((5 + 2) * (4 - 3) / 2)

    # Verify gradients
    result.backward()

    # d/da = (4-b)/2 = (4-3)/2 = 0.5
    assert a.grad == pytest.approx(0.5)

    # d/db = -(5+a)/2 = -(5+2)/2 = -3.5
    assert b.grad == pytest.approx(-3.5)


def test_reverse_operations_with_non_float32():
    """test reverse operations with Python native types"""
    a = Float32(3.0, requires_grad=True)

    # test with integer
    result_int = 5 + a
    assert isinstance(result_int, Float32)
    assert result_int.item() == 8.0

    # test with float
    result_float = 2.5 * a
    assert isinstance(result_float, Float32)
    assert result_float.item() == 7.5


@pytest.mark.parametrize("base,exp", [
    (2.0, 3.0),  # d/dx (x^3) = 3x^2
    (3.0, 2.0),  # d/dx (x^2) = 2x
    (4.0, 0.5),  # d/dx (sqrt(x)) = 0.5/sqrt(x)
])
def test_pow_gradient_base(base, exp):
    """test gradient calculation for base in a**b"""
    a = Float32(base, requires_grad=True)
    result = a ** exp
    result.backward()

    expected_grad = exp * (base ** (exp - 1))
    assert a.grad == pytest.approx(expected_grad)


@pytest.mark.parametrize("base,exp", [
    (2.0, 3.0),  # d/dx (2^x) = 2^x * ln(2)
    (3.0, 2.0),  # d/dx (3^x) = 3^x * ln(3)
])
def test_pow_gradient_exp(base, exp):
    """test gradient calculation for exponent in a**b"""
    b = Float32(exp, requires_grad=True)
    result = base ** b
    result.backward()

    expected_grad = (base ** exp) * math.log(base)
    assert b.grad == pytest.approx(expected_grad)


def test_combined_gradients():
    """test gradients when both base and exponent require grad"""
    a = Float32(2.0, requires_grad=True)
    b = Float32(3.0, requires_grad=True)
    result = a ** b
    result.backward()

    # Gradient for base: 3 * 2^2 = 12
    assert a.grad == pytest.approx(12.0)
    # Gradient for exponent: 8 * ln(2) ≈ 5.545
    assert b.grad == pytest.approx((2 ** 3) * math.log(2))


def test_zero_base_gradient():
    """test gradient with zero base (0^x special case)"""
    a = Float32(0.0, requires_grad=True)
    result = a ** 2.0
    result.backward()
    assert a.grad == 0.0  # Derivative of 0^x at 0 is 0


def test_one_base_gradient():
    """test gradient when base is 1 (1^x special case)"""
    a = Float32(1.0, requires_grad=True)
    b = Float32(3.0, requires_grad=True)
    result = a ** b
    result.backward()

    # Gradient for base: 3 * 1^2 = 3
    assert a.grad == pytest.approx(3.0)
    # Gradient for exponent: 1^x * ln(1) = 0
    assert b.grad == pytest.approx(0.0)


def test_rpow_gradient():
    """test gradient for reverse power operation"""
    a = Float32(3.0, requires_grad=True)
    result = 2.0 ** a  # Calls __rpow__
    result.backward()

    # Expected gradient: 2^3 * ln(2) ≈ 8 * 0.693 ≈ 5.545
    assert a.grad == pytest.approx((2 ** 3) * math.log(2))


def test_detach_returns_new_object():
    """Test that detach() returns a new object with same value"""
    original = Float32(5.0, requires_grad=True)
    detached = original.detach()

    assert detached.item() == original.item()
    assert detached is not original  # Must be a new object


def test_detach_breaks_gradient_connection():
    """Test that detach() breaks gradient computation chain"""
    a = Float32(2.0, requires_grad=True)
    b = a.detach()  # Detached copy
    c = Float32(3.0, requires_grad=True)
    result = b * c  # Operation with detached tensor

    result.backward()

    # Verify gradients
    assert c.grad == pytest.approx(2.0)  # dc = b = 2.0
    assert a.grad is None  # Detached tensor shouldn't affect original


def test_detach_with_computational_graph():
    """Test detach() in middle of computational graph"""
    a = Float32(2.0, requires_grad=True)
    b = Float32(3.0, requires_grad=True)
    c = a * b
    d = c.detach()  # Detach here
    e = d * Float32(4.0, requires_grad=True)

    e.backward()

    assert e.grad is None
    assert c.grad is None
    assert b.grad is None  # Detached before reaching b
    assert a.grad is None  # Detached before reaching a
    assert e.item() == pytest.approx(24.0)  # 2*3*4=24


def test_detach_then_reattach():
    """Test that detached tensor can be reattached to graph"""
    a = Float32(2.0, requires_grad=True)
    b = a.detach()
    c = Float32(b.item(), requires_grad=True)  # Reattach

    result = c * 3.0
    result.backward()

    assert c.grad == pytest.approx(3.0)
    assert a.grad is None  # Original remains unaffected


def test_detach_multiple_calls():
    """Test multiple detach() calls don't affect behavior"""
    a = Float32(3.0, requires_grad=True)
    b = a.detach().detach().detach()

    assert b.item() == pytest.approx(3.0)
    assert b is not a.detach()  # Each detach creates new object


def test_multiple_usage():
    "Test the case where single object is used twice"
    a = Float32(2, requires_grad=True)

    b = Float32(1, requires_grad=True)

    c = a + b

    m = c * a

    m.backward()

    assert a.grad == 5
    assert b.grad == 2


def test_unused_grad():
    a = Float32(2, requires_grad=True)

    b = Float32(1, requires_grad=True)

    c = a + b

    m = c * a

    k = m ** 2

    k.backward()

    assert m.grad is None
    assert a.grad == 60
    assert b.grad == 24


def test_single_scalar_grad():
    a = Float32(2, requires_grad=True)

    c = a + a

    m = c * a

    k = m ** 2

    k.backward()

    assert a.grad == 128


def test_backward_without_retain_graph():
    """
    Test that backward without retain_graph frees the graph and subsequent backward
    either raises an error or does not accumulate again.
    """
    # Simple expression: (a + b)^2
    a = Float32(2.0, requires_grad=True)
    b = Float32(3.0, requires_grad=True)
    c = a + b
    d = c * c  # d = (a + b)^2

    d.backward()

    # Second backward without retain_graph
    with pytest.raises(RuntimeError) as e:
        d.backward()

    assert str(e.value) == "The computational graph was cleaned up after the backward"


def test_backward_with_retain_graph():
    """
    Test that backward with retain_graph=True allows multiple backward passes
    and accumulates gradients correctly.
    """
    # Simple expression: a * b
    a = Float32(4.0, requires_grad=True)
    b = Float32(5.0, requires_grad=True)
    c = a * b

    # 1st pass
    c.backward(retain_graph=True)
    assert a.grad == pytest.approx(5.0)
    assert b.grad == pytest.approx(4.0)

    # 2nd pass
    c.backward(retain_graph=True)
    assert a.grad == pytest.approx(2 * 5.0)
    assert b.grad == pytest.approx(2 * 4.0)

    # Final pass without retain_graph
    c.backward()
    assert a.grad == pytest.approx(3 * 5.0)
    assert b.grad == pytest.approx(3 * 4.0)

    with pytest.raises(RuntimeError) as e:
        c.backward()

    assert str(e.value) == "The computational graph was cleaned up after the backward"
