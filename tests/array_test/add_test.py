import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from tests import TEST_VALUES_BASE, compare_tensor_ops
from src import Array


# -------- Base tests --------

@pytest.mark.parametrize(
    "x1, x2",
    TEST_VALUES_BASE
)
def test_result_eq(x1, x2):
    x1 = np.array(x1).astype(np.float32)
    x2 = np.array(x2).astype(np.float32)

    x1_arr = Array(x1, dtype=np.float32)
    x2_arr = Array(x2, dtype=np.float32)

    x_sum = x1 + x2
    x_sum_arr = x1_arr + x2_arr

    assert np.allclose(x_sum, x_sum_arr.data)
    assert not x_sum_arr.requires_grad
    assert x_sum_arr._grad_fn is None


@pytest.mark.parametrize(
    "x1, x2",
    TEST_VALUES_BASE
)
def test_result_eq_with_numpy(x1, x2):
    x1 = np.array(x1).astype(np.float32)
    x2 = np.array(x2).astype(np.float32)

    x2_arr = Array(x2, dtype=np.float32, requires_grad=True)

    x_sum = x1 + x2_arr

    assert np.allclose(x_sum.data, x1 + x2)
    assert not x_sum.requires_grad
    assert x_sum._grad_fn is not None


@pytest.mark.parametrize(
    "x1, x2",
    TEST_VALUES_BASE
)
def test_requires_grad1(x1, x2):
    x1 = np.array(x1).astype(np.float32)
    x2 = np.array(x2).astype(np.float32)

    x1_arr = Array(x1, dtype=np.float32, requires_grad=True)
    x2_arr = Array(x2, dtype=np.float32)

    x_sum = x1_arr + x2_arr

    assert not x_sum.requires_grad
    assert x_sum._grad_fn is not None


@pytest.mark.parametrize(
    "x1, x2",
    TEST_VALUES_BASE
)
def test_requires_grad2(x1, x2):
    x1 = np.array(x1).astype(np.float32)
    x2 = np.array(x2).astype(np.float32)

    x1_arr = Array(x1, dtype=np.float32)
    x2_arr = Array(x2, dtype=np.float32, requires_grad=True)

    x_sum = x1_arr + x2_arr

    assert not x_sum.requires_grad
    assert x_sum._grad_fn is not None


# -------- Comparison with torch --------

@pytest.mark.parametrize(
    "x1, x2",
    TEST_VALUES_BASE
)
def test_add_compat(x1, x2):
    x1 = np.array(x1).astype(np.float32)
    x2 = np.array(x2).astype(np.float32)

    def fn_f():
        a = Array(x1, requires_grad=True)
        b = Array(x2, requires_grad=True)
        return a + b, [a, b]
    def fn_t():
        a = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
        return a + b, [a, b]
    compare_tensor_ops(fn_f, fn_t, name="add")
