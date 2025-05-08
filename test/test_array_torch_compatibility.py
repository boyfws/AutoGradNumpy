import numpy as np
import torch
import pytest

from add_src_to_path import append_src
append_src()

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

    assert arr_res.requires_grad
    assert not arr_res.is_leaf

    arr_out = arr_res.data

    # torch
    t_res, t_inputs = fn_torch()
    if isinstance(t_res, torch.Tensor):
        t_res.sum().backward()
    else:
        t_res.backward()

    assert t_res.requires_grad
    assert not t_res.is_leaf

    t_out = t_res.detach().cpu().numpy()

    arr_grads = []
    t_grads = []
    for i in range(len(t_inputs)):
        assert t_inputs[i].is_leaf
        assert arr_inputs[i].is_leaf
        if arr_inputs[i].grad is not None and t_inputs[i].grad is not None:
            arr_grads.append(arr_inputs[i].grad)
            t_grads.append(t_inputs[i].grad.cpu().numpy())

    assert len(arr_grads) > 0
    # compare outputs
    assert np.allclose(arr_out, t_out), f"Output mismatch in {name}: {arr_out} vs {t_out}"
    # compare grads
    for i, (ag, tg) in enumerate(zip(arr_grads, t_grads)):
        assert np.allclose(ag, tg), f"Grad #{i} mismatch in {name}: {ag} vs {tg}"


@pytest.mark.parametrize("shape", [(2, 3), (4,)])
def test_array_add_compat(shape):
    a_np = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    def fn_arr():
        a = Array(
            a_np,
            requires_grad=True
        )
        b = Array(
            np.ones(shape, dtype=float),
            requires_grad=True
        )
        return a + b, [a, b]

    def fn_t():
        a = torch.tensor(a_np, requires_grad=True)
        b = torch.ones(shape, dtype=torch.float32, requires_grad=True)
        return a + b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="array_add")


@pytest.mark.parametrize("shape", [(2, 3), (4,)])
def test_array_sub_compat(shape):
    array = np.arange(np.prod(shape)).reshape(shape).astype(float)

    def fn_arr():
        a = Array(array, requires_grad=True)
        b = Array(np.ones(shape, dtype=float), requires_grad=True)
        return a - b, [a, b]

    def fn_t():
        a = torch.tensor(array.astype(np.float32), requires_grad=True)
        b = torch.ones(shape, dtype=torch.float32, requires_grad=True)
        return a - b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="array_sub")


@pytest.mark.parametrize("shape", [(2, 3), (4,)])
def test_array_mul_compat(shape):
    array = np.arange(np.prod(shape)).reshape(shape).astype(float)

    def fn_arr():
        a = Array(array, requires_grad=True)
        b = Array(np.full(shape, 2.5), requires_grad=True)
        return a * b, [a, b]

    def fn_t():
        a = torch.tensor(array.astype(np.float32), requires_grad=True)
        b = torch.full(shape, 2.5, dtype=torch.float32, requires_grad=True)
        return a * b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="array_mul")


@pytest.mark.parametrize("shape", [(2, 3), (4,)])
def test_array_div_compat(shape):
    array = np.arange(1, np.prod(shape) + 1).reshape(shape).astype(float)

    def fn_arr():
        a = Array(array, requires_grad=True)
        b = Array(np.full(shape, 3.0), requires_grad=True)
        return a / b, [a, b]

    def fn_t():
        a = torch.tensor(array.astype(np.float32), requires_grad=True)
        b = torch.full(shape, 3.0, dtype=torch.float32, requires_grad=True)
        return a / b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="array_div")


@pytest.mark.parametrize("shape,axis", [((2, 3), None), ((2, 3), 0), ((2, 3), 1)])
def test_array_sum_compat(shape, axis):
    a_np = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)

    def fn_arr():
        a = Array(
            a_np,
            requires_grad=True
        )
        return (a.sum() if axis is None else a.sum(axis=axis)), [a]

    def fn_t():
        a = torch.tensor(a_np, requires_grad=True)
        return (a.sum() if axis is None else a.sum(dim=axis)), [a]

    compare_tensor_ops(fn_arr, fn_t, name=f"array_sum_axis{axis}")


def test_array_neg_compat():
    def fn_arr():
        a = Array(np.array([1.0, -1.0, 2.0]), requires_grad=True)
        return -a, [a]

    def fn_t():
        a = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32, requires_grad=True)
        return -a, [a]

    compare_tensor_ops(fn_arr, fn_t, name="array_neg")


def test_broadcast1():
    array_a = np.array([1, 2, 3])
    array_b = np.array([
        [0, 8, 7],
        [8, 9, 2]
    ])

    def fn_arr():
        a = Array(array_a, requires_grad=True)
        b = Array(array_b, requires_grad=True)
        return a + b, [a, b]

    def fn_t():
        a = torch.tensor(array_a.astype(np.float32), requires_grad=True)
        b = torch.tensor(array_b.astype(np.float32), requires_grad=True)
        return a + b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="array_broadcast1")


def test_broadcast2():
    array_a = np.array([1, 2, 3])
    array_b = np.array([
        [0, 8, 7],
        [8, 9, 2]
    ])

    def fn_arr():
        a = Array(array_a, requires_grad=True)
        b = Array(array_b, requires_grad=True)
        return (a + b) * b, [a, b]

    def fn_t():
        a = torch.tensor(array_a.astype(np.float32), requires_grad=True)
        b = torch.tensor(array_b.astype(np.float32), requires_grad=True)
        return (a + b) * b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="array_broadcast2")


TEST_ARRAYS = [
    (
        np.array([0, 8, 5]),
        np.array([1, 2, 3]),
    ),
    (
        np.array(
            [
                [
                    [1, 2],
                    [2, 1]
                ],
                [
                    [2, 1],
                    [1, 0]
                ],
                [
                    [0, 0],
                    [10, 10]
                ]
            ]
        ),
        np.array([1, 2, 3]).reshape(3, 1, 1),
    ),
    (
        np.array(
            [
                [
                    [1, 2],
                    [2, 1]
                ],
                [
                    [2, 1],
                    [1, 0]
                ],
                [
                    [0, 0],
                    [10, 10]
                ]
            ]
        ),
        np.array([
            [3, 1],
            [2, 2]
        ]),
    )
]


@pytest.mark.parametrize("array_a, array_b", TEST_ARRAYS)
def test_pow(array_a, array_b):
    def fn_arr():
        a = Array(array_a, requires_grad=True)
        b = Array(array_b, requires_grad=True)
        return (a ** b) * b, [a, b]

    def fn_t():
        a = torch.tensor(array_a.astype(np.float32), requires_grad=True)
        b = torch.tensor(array_b.astype(np.float32), requires_grad=True)
        return (a ** b) * b, [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="test_pow")


@pytest.mark.parametrize("array_a, array_b", TEST_ARRAYS)
def test_truediv(array_a, array_b):
    def fn_arr():
        a = Array(array_a, requires_grad=True)
        b = Array(array_b, requires_grad=True)
        return (a / b) * (b - a), [a, b]

    def fn_t():
        a = torch.tensor(array_a.astype(np.float32), requires_grad=True)
        b = torch.tensor(array_b.astype(np.float32), requires_grad=True)
        return (a / b) * (b - a), [a, b]

    compare_tensor_ops(fn_arr, fn_t, name="test_truediv")


def test_rpow():
    array = np.array([1, 2, 4]).astype(np.float32)

    def fn_arr():
        a = Array(array, requires_grad=True)
        c = 2 ** a
        return c, [a]

    def fn_t():
        a = torch.tensor(array, requires_grad=True)
        c = 2 ** a
        return c, [a]

    compare_tensor_ops(fn_arr, fn_t, name="rpow")


def test_rtruediv():
    array = np.array([1, 2, 4]).astype(np.float32)

    def fn_arr():
        a = Array(array, requires_grad=True)
        c = 2 / a
        return c, [a]

    def fn_t():
        a = torch.tensor(array, requires_grad=True)
        c = 2 / a
        return c, [a]

    compare_tensor_ops(fn_arr, fn_t, name="rpow")


@pytest.mark.parametrize("req_g1, req_g2", [
    (True, False),
    (False, True),
    (True, True),
    (False, False),
])
def test_requires_grad(req_g1, req_g2):
    array1 = np.array([1, 2, 4]).astype(np.float32)
    array2 = np.array([2, 9, 9]).astype(np.float32)

    a_ar = Array(array1, requires_grad=req_g1)
    b_ar = Array(array2, requires_grad=req_g2)
    c_ar = a_ar + b_ar

    a_t = torch.tensor(array1, requires_grad=req_g1)
    b_t = torch.tensor(array2, requires_grad=req_g2)
    c_t = a_t + b_t

    assert c_t.requires_grad == c_ar.requires_grad
    assert c_t.requires_grad == (req_g1 or req_g2)
    assert c_t.is_leaf == c_ar.is_leaf
    assert c_t.is_leaf == (not (req_g1 or req_g2))
    assert a_t.is_leaf == a_ar.is_leaf == True


@pytest.mark.parametrize("req_g1", [
    True,
    False,
])
def test_requires_grad2(req_g1):
    array1 = np.array([1, 2, 4]).astype(np.float32)

    a_ar = Array(array1, requires_grad=req_g1)
    c_ar = a_ar + 2

    a_t = torch.tensor(array1, requires_grad=req_g1)
    c_t = a_t + 2

    assert c_t.requires_grad == c_ar.requires_grad
    assert c_t.requires_grad == req_g1
    assert c_t.is_leaf == c_ar.is_leaf
    assert c_t.is_leaf == (not req_g1)
    assert a_t.is_leaf == a_ar.is_leaf
    assert a_t.is_leaf == True


