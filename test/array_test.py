import numpy as np
import pytest
from src import Array


@pytest.mark.parametrize("dtype", [
    np.float32,
    np.float64,
    np.float16
]
                         )
def test_init(dtype) -> None:
    array = np.array([1, 2, 3], dtype=np.int32)
    array_obj = Array(array, dtype=dtype)

    assert all(array_obj.item() == array)
    assert array_obj.item().dtype == dtype


@pytest.mark.parametrize("dtype", [
    np.int32,
    np.complex64,
    np.int64
])
def test_init2(dtype) -> None:
    array = np.array([1, 2, 3], dtype=np.float32)
    with pytest.raises(TypeError):
        _ = Array(array, dtype=dtype)


def test_init3():
    array = np.array([1, 2, 3], dtype=np.int32)
    ar = Array(array)

    assert ar.item().dtype == np.float32
    assert all(
        ar.item() == array.astype(np.float32)
    )


def get_item_test() -> None:
    array = np.array([
        [1, 2, 3],
        [1, 4, 9]
    ], dtype=np.float32)

    ar_obj = Array(array)

    assert ar_obj[:3] == array[:3]
    assert ar_obj[0] == array[0]
    assert ar_obj[1, 2] == array[1, 2]
    assert ar_obj[:1, 2] == array[:1, 2]


def test_setitem_single_element():
    """Test setting a single element by index"""
    arr = Array(
        np.array([1, 2, 3])
    )
    arr[1] = 5
    assert arr.item().tolist() == [1, 5, 3]


def test_setitem_slice():
    """Test setting values using slice notation"""
    arr = Array([[1, 2], [3, 4]])
    arr[:, 0] = [10, 20]
    assert arr.data.tolist() == [[10, 2], [20, 4]]

def test_setitem_boolean_mask():
    """Test setting values using boolean masking"""
    arr = Array([1, 2, 3, 4])
    arr[arr.data > 2] = 0
    assert arr.data.tolist() == [1, 2, 0, 0]

def test_setitem_resets_grad():
    """Test that gradients are reset for modified elements"""
    arr = Array([1.0, 2.0, 3.0], requires_grad=True)
    arr.sum().backward()  # Original grad [1, 1, 1]
    arr[1] = 10.0
    assert arr.grad.tolist() == [1.0, 0.0, 1.0]

def test_setitem_whole_tensor_resets_grad():
    """Test that modifying entire tensor resets all gradients"""
    arr = Array([1.0, 2.0], requires_grad=True)
    arr[:] = [3.0, 4.0]
    arr.sum().backward()
    assert arr.grad.tolist() == [1.0, 1.0]

def test_setitem_with_requires_grad():
    """Test modification when requires_grad=True"""
    arr = Array([1.0, 2.0], requires_grad=True)
    arr[0] = 5.0  # Should not raise error
    assert arr.data[0] == 5.0

def test_setitem_changes_shape():
    """Test that shape-changing operations raise error"""
    arr = Array([1, 2, 3])
    with pytest.raises(ValueError):
        arr[:2] = [10, 20, 30]  # Shape mismatch

def test_setitem_chain_operations():
    """Test modification during computational graph"""
    a = Array([1.0, 2.0], requires_grad=True)
    b = a * 2  # [2.0, 4.0]
    b[1] = 10.0  # [2.0, 10.0]
    loss = b.sum()
    loss.backward()
    assert a.grad.tolist() == [1.0, 0.0]

def test_setitem_non_contiguous():
    """Test advanced indexing scenarios"""
    arr = Array([[1, 2], [3, 4]])
    arr[[0, 1], [1, 0]] = [10, 20]
    assert arr.data.tolist() == [[1, 10], [20, 4]]

def test_setitem_after_math_operations():
    """Test setitem after mathematical operations"""
    arr = Array([1.0, 2.0], requires_grad=True)
    arr = arr * 2  # [2.0, 4.0]
    arr[0] = 5.0
    arr.sum().backward()
    assert hasattr(arr, 'grad')

def test_setitem_with_custom_types():
    """Test setting non-numeric values"""
    arr = Array(["a", "b", "c"])
    arr[1] = "x"
    assert arr.data.tolist() == ["a", "x", "c"]

def test_setitem_multidimensional():
    """Test setting values in multidimensional arrays"""
    arr = Array(np.zeros((3, 3)))
    arr[1, 1] = 5
    assert arr.data[1, 1] == 5

def test_setitem_with_step_slice():
    """Test setting with step slices"""
    arr = Array([0, 1, 2, 3, 4, 5])
    arr[::2] = [10, 20, 30]
    assert arr.data.tolist() == [10, 1, 20, 3, 30, 5]

def test_setitem_broadcasting():
    """Test value broadcasting during assignment"""
    arr = Array(np.zeros((2, 3)))
    arr[:, 1] = 5
    assert np.all(arr.data[:, 1] == 5)