import numpy as np

from add_src_to_path import append_src
append_src()

from src import Array


def test_broadcast_add_grad_array():
    # shapes (2,1) + (2,3)
    a = Array(np.array([[1.0],[2.0]]), requires_grad=True)
    b = Array(np.array([[10.0,20.0,30.0],[40.0,50.0,60.0]]), requires_grad=True)
    c = a + b
    c.sum().backward()
    exp_a = np.sum(np.ones((2,3)), axis=1, keepdims=True)
    assert np.allclose(a.grad, exp_a)
    assert np.allclose(b.grad, np.ones_like(c.data))


def test_broadcast_sub_grad_array():
    # shapes (2,1) - (2,3)
    a = Array(np.array([[1.0],[2.0]]), requires_grad=True)
    b = Array(np.array([[5.0,6.0,7.0],[8.0,9.0,10.0]]), requires_grad=True)
    c = a - b
    c.sum().backward()
    exp_a = np.sum(np.ones((2,3)), axis=1, keepdims=True)
    assert np.allclose(a.grad, exp_a)
    assert np.allclose(b.grad, -np.ones_like(c.data))


def test_negation_grad_array():
    x = Array(np.array([1.0, -2.0, 3.0]), requires_grad=True)
    y = -x
    y.sum().backward()
    assert np.allclose(x.grad, -np.ones_like(y.data))


def test_sum_no_axis_grad_array():
    x = Array(np.array([[1.0,2.0],[3.0,4.0]]), requires_grad=True)
    s = x.sum()
    s.backward()
    assert np.allclose(x.grad, np.ones_like(x.data))


def test_sum_with_axis_grad_array():
    x = Array(np.array([[1.0,2.0],[3.0,4.0]]), requires_grad=True)
    s = x.sum(axis=0)
    grad_out = np.array([1.0,1.0])
    s.sum().backward()
    expected = np.broadcast_to(grad_out.reshape(1,-1), x.data.shape)
    assert np.allclose(x.grad, expected)


def test_repeat_backward_array():
    x = Array(np.array([[1.0,2.0],[3.0,4.0]]), requires_grad=True)
    s = x.sum()
    s.backward()
    first = x.grad.copy()
    x._zero_grad()
    try:
        s.backward()
        assert np.allclose(x.grad, first)
    except RuntimeError:
        pass
