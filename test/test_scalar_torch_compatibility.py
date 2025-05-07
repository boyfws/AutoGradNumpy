import pytest
import torch

from add_src_to_path import append_src
append_src()

from src import Float32

# Test values (integers to avoid precision issues)
TEST_VALUES = [(10, 3), (100, 20), (15, 5)]
TEST_VALUES_DIV = [(10, 2), (100, 25), (15, 3)]  # avoid zero
POW_TEST_VALUES = [(2.0, 3.0), (3.0, 2.0), (4.0, 0.5)]  # (base, exp)


def compare_scalar_ops(fn_float, fn_torch, name=""):
    """
    Helper: compare outputs and gradients for a scalar op.
    """
    # Float32
    f_res, f_inputs = fn_float()
    f_res.backward()
    f_out = f_res.item()
    f_grads = [inp.grad for inp in f_inputs]

    # torch
    t_res, t_inputs = fn_torch()
    t_res.backward()
    t_out = t_res.item()
    t_grads = [inp.grad.item() for inp in t_inputs]

    # compare outputs
    assert f_out == pytest.approx(t_out), f"Output mismatch in {name}: {f_out} vs {t_out}"
    # compare grads
    for i, (fg, tg) in enumerate(zip(f_grads, t_grads)):
        assert fg == pytest.approx(tg), f"Grad #{i} mismatch in {name}: {fg} vs {tg}"


@pytest.mark.parametrize("x1, x2", TEST_VALUES)
def test_add_compat(x1, x2):
    def fn_f():
        a = Float32(x1, requires_grad=True)
        b = Float32(x2, requires_grad=True)
        return a + b, [a, b]
    def fn_t():
        a = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
        return a + b, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="add")


@pytest.mark.parametrize("x1, x2", TEST_VALUES)
def test_sub_compat(x1, x2):
    def fn_f():
        a = Float32(x1, requires_grad=True)
        b = Float32(x2, requires_grad=True)
        return a - b, [a, b]
    def fn_t():
        a = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
        return a - b, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="sub")


@pytest.mark.parametrize("x1, x2", TEST_VALUES)
def test_mul_compat(x1, x2):
    def fn_f():
        a = Float32(x1, requires_grad=True)
        b = Float32(x2, requires_grad=True)
        return a * b, [a, b]
    def fn_t():
        a = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
        return a * b, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="mul")


@pytest.mark.parametrize("x1, x2", TEST_VALUES_DIV)
def test_div_compat(x1, x2):
    def fn_f():
        a = Float32(x1, requires_grad=True)
        b = Float32(x2, requires_grad=True)
        return a / b, [a, b]
    def fn_t():
        a = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
        return a / b, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="div")


def test_negation_compat():
    def fn_f():
        a = Float32(3.0, requires_grad=True)
        return -a, [a]
    def fn_t():
        a = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
        return -a, [a]
    compare_scalar_ops(fn_f, fn_t, name="neg")


@pytest.mark.parametrize("base,exp", POW_TEST_VALUES)
def test_pow_const_exp_compat(base, exp):
    def fn_f():
        a = Float32(base, requires_grad=True)
        return a ** exp, [a]
    def fn_t():
        a = torch.tensor(base, dtype=torch.float32, requires_grad=True)
        return a ** exp, [a]
    compare_scalar_ops(fn_f, fn_t, name="pow_const_exp")


@pytest.mark.parametrize("base,exp", POW_TEST_VALUES[:2])  # integer exponents
def test_pow_const_base_compat(base, exp):
    def fn_f():
        b = Float32(exp, requires_grad=True)
        return base ** b, [b]
    def fn_t():
        b = torch.tensor(exp, dtype=torch.float32, requires_grad=True)
        return base ** b, [b]
    compare_scalar_ops(fn_f, fn_t, name="pow_const_base")


def test_pow_both_compat():
    def fn_f():
        a = Float32(2.0, requires_grad=True)
        b = Float32(3.0, requires_grad=True)
        return a ** b, [a, b]
    def fn_t():
        a = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
        return a ** b, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="pow_both")


def test_complex_composite_compat():
    def fn_f():
        a = Float32(2.5, requires_grad=True)
        b = Float32(4.0, requires_grad=True)
        expr = (a + b) * (-a) / (b - 1)
        return expr, [a, b]
    def fn_t():
        a = torch.tensor(2.5, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(4.0, dtype=torch.float32, requires_grad=True)
        expr = (a + b) * (-a) / (b - 1)
        return expr, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="complex")


def test_nested_composite_compat():
    def fn_f():
        a = Float32(1.5, requires_grad=True)
        b = Float32(2.5, requires_grad=True)
        expr = ((a * b) + (a / b)) - (b - a)
        return expr, [a, b]
    def fn_t():
        a = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(2.5, dtype=torch.float32, requires_grad=True)
        expr = ((a * b) + (a / b)) - (b - a)
        return expr, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="nested_composite")


def test_chained_ops_compat():
    def fn_f():
        a = Float32(3.0, requires_grad=True)
        b = Float32(5.0, requires_grad=True)
        expr = (a + b) * (a - b) / (a * 2.0 + b / 2.0)
        return expr, [a, b]
    def fn_t():
        a = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
        expr = (a + b) * (a - b) / (a * 2.0 + b / 2.0)
        return expr, [a, b]
    compare_scalar_ops(fn_f, fn_t, name="chained_ops")