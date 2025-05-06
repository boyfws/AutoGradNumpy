import numpy as np
import torch
import pytest
from src import Array


def compare_tensor_ops(fn_arr, fn_torch, name=""):
    """
    Helper: compare outputs and gradients for Array vs torch.Tensor ops.
    """
    # Array
    arr_res, arr_inputs = fn_arr()
    # if output is scalar, use scalar grad, else ones_like
    if isinstance(arr_res, Array):
        arr_res.sum().backward()
    else:
        arr_res.backward()

    arr_out = arr_res.data

    # torch
    t_res, t_inputs = fn_torch()
    if isinstance(t_res, torch.Tensor):
        t_res.sum().backward()
    else:
        t_res.backward()

    t_out = t_res.detach().cpu().numpy()

    arr_grads = []
    t_grads = []
    for i in range(len(t_inputs)):
        if arr_inputs[i].grad is not None and t_inputs[i].grad is not None:
            arr_grads.append(arr_inputs[i].grad)
            t_grads.append(t_inputs[i].grad.cpu().numpy())


    # compare outputs
    assert np.allclose(arr_out, t_out), f"Output mismatch in {name}: {arr_out} vs {t_out}"
    # compare grads
    for i, (ag, tg) in enumerate(zip(arr_grads, t_grads)):
        assert np.allclose(ag, tg), f"Grad #{i} mismatch in {name}: {ag} vs {tg}"


@pytest.mark.parametrize("shape", [(2,3), (4,)] )
def test_array_add_compat(shape):
    def fn_arr():
        a = Array(np.arange(np.prod(shape)).reshape(shape).astype(float), requires_grad=True)
        b = Array(np.ones(shape, dtype=float), requires_grad=True)
        return a + b, [a, b]
    def fn_t():
        a = torch.arange(np.prod(shape), dtype=torch.float32, requires_grad=True).reshape(shape)
        b = torch.ones(shape, dtype=torch.float32, requires_grad=True)
        return a + b, [a, b]
    compare_tensor_ops(fn_arr, fn_t, name="array_add")

@pytest.mark.parametrize("shape", [(2,3), (4,)] )
def test_array_sub_compat(shape):
    array = np.arange(np.prod(shape)).reshape(shape).astype(float)
    def fn_arr():
        a = Array(array, requires_grad=True)
        b = Array(np.ones(shape, dtype=float), requires_grad=True)
        return a - b, [a, b]

    def fn_t():
        a = torch.Tensor(array).to(torch.float32)
        a.requires_grad=True
        b = torch.ones(shape, dtype=torch.float32, requires_grad=True)
        return a - b, [a, b]
    compare_tensor_ops(fn_arr, fn_t, name="array_sub")

@pytest.mark.parametrize("shape", [(2,3), (4,)] )
def test_array_mul_compat(shape):
    array = np.arange(np.prod(shape)).reshape(shape).astype(float)

    def fn_arr():
        a = Array(array, requires_grad=True)
        b = Array(np.full(shape, 2.5), requires_grad=True)
        return a * b, [a, b]

    def fn_t():
        a = torch.Tensor(array).to(torch.float32)
        a.requires_grad= True
        b = torch.full(shape, 2.5, dtype=torch.float32, requires_grad=True)
        return a * b, [a, b]
    compare_tensor_ops(fn_arr, fn_t, name="array_mul")


@pytest.mark.parametrize("shape", [(2,3), (4,)] )
def test_array_div_compat(shape):
    array = np.arange(1, np.prod(shape)+1).reshape(shape).astype(float)
    def fn_arr():
        a = Array(array, requires_grad=True)
        b = Array(np.full(shape, 3.0), requires_grad=True)
        return a / b, [a, b]
    def fn_t():
        a = torch.Tensor(array).to(torch.float32)
        a.requires_grad=True
        b = torch.full(shape, 3.0, dtype=torch.float32, requires_grad=True)
        return a / b, [a, b]
    compare_tensor_ops(fn_arr, fn_t, name="array_div")


@pytest.mark.parametrize("shape,axis", [((2,3), None), ((2,3), 0), ((2,3), 1)])
def test_array_sum_compat(shape, axis):
    def fn_arr():
        a = Array(np.arange(np.prod(shape)).reshape(shape).astype(float), requires_grad=True)
        return a.sum() if axis is None else a.sum(axis=axis), [a]
    def fn_t():
        a = torch.arange(np.prod(shape), dtype=torch.float32, requires_grad=True).reshape(shape)
        return a.sum() if axis is None else a.sum(dim=axis), [a]
    compare_tensor_ops(fn_arr, fn_t, name=f"array_sum_axis{axis}")


def test_array_neg_compat():
    def fn_arr():
        a = Array(np.array([1.0, -1.0, 2.0]), requires_grad=True)
        return -a, [a]
    def fn_t():
        a = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32, requires_grad=True)
        return -a, [a]
    compare_tensor_ops(fn_arr, fn_t, name="array_neg")