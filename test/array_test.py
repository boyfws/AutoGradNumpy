import numpy as np
import pytest

from add_src_to_path import append_src
append_src()

from src import Array, Float32

@pytest.mark.parametrize("dtype", [
    np.float32,
    np.float64,
    np.float16
])
def test_init(dtype) -> None:
    array = np.array([1, 2, 3], dtype=np.int32)
    array_obj = Array(array, dtype=dtype)

    assert all(array_obj.data == array)
    assert array_obj.data.dtype == dtype


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

    assert ar.data.dtype == np.float32
    assert all(
        ar.data == array.astype(np.float32)
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
    assert arr.data.tolist() == [1, 5, 3]


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



def test_setitem_with_requires_grad():
    """Test modification when requires_grad=True"""
    arr = Array([1.0, 2.0], requires_grad=True)
    arr[0] = 5.0  # Should not raise error
    assert arr.data[0] == 5.0
    assert not arr.requires_grad


def test_setitem_changes_shape():
    """Test that shape-changing operations raise error"""
    arr = Array([1, 2, 3])
    with pytest.raises(ValueError):
        arr[:2] = [10, 20, 30]  # Shape mismatch


def test_setitem_non_contiguous():
    """Test advanced indexing scenarios"""
    arr = Array([[1, 2], [3, 4]])
    arr[[0, 1], [1, 0]] = [10, 20]
    assert arr.data.tolist() == [[1, 10], [20, 4]]


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


TEST_VALUES = [
    (
            np.array([1, 2, 3]),
            np.array([1, 2, 3])
    ),
    (
            np.array(
                [
                    [1, 2, 3],
                    [1, 9, 9]
                ]
            ),
            np.array([9, 9, 20])
    ),
    (
            np.array(
                [
                    [1, 2],
                    [3,  7]
                ]
            ),
            np.array([
                [1],
                [2]
            ])
    ),
    (
            np.array(
                [
                    [1, 2, 4],
                    [9, 8, 10]
                ]
            ),
            np.float32(2)
    ),
    (
            np.float32(5),
            np.array([1, 8, 4])
    )
]

TEST_DTYPES = [
        (np.float16, np.float16, np.float16),
        (np.float16, np.float32, np.float32),
        (np.float32, np.float32, np.float32),
        (np.float32, np.float64, np.float64),
        (np.float64, np.float64, np.float64),
]


@pytest.mark.parametrize("a, b", TEST_VALUES)
def test_add(a, b):
    if isinstance(a, np.ndarray):
        a_trans = Array(a)
    else:
        a_trans = Float32(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Float32(b)

    res_1 = a + b_trans
    res_2 = a_trans + b
    res_3 = a_trans + b_trans

    for el in (
        res_1,
        res_2,
        res_3
    ):
        assert isinstance(el, Array)
        assert np.allclose(el.data, a + b)


@pytest.mark.parametrize("a, b", TEST_VALUES)
def test_sub(a, b):
    if isinstance(a, np.ndarray):
        a_trans = Array(a)
    else:
        a_trans = Float32(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Float32(b)

    res_1 = a - b_trans
    res_2 = a_trans - b
    res_3 = a_trans - b_trans

    for el in (
            res_1,
            res_2,
            res_3
    ):
        assert isinstance(el, Array)
        assert np.allclose(el.data, a - b)


@pytest.mark.parametrize("a, b", TEST_VALUES)
def test_mul(a, b):
    if isinstance(a, np.ndarray):
        a_trans = Array(a)
    else:
        a_trans = Float32(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Float32(b)

    res_1 = a * b_trans
    res_2 = a_trans * b
    res_3 = a_trans * b_trans

    for el in (
            res_1,
            res_2,
            res_3
    ):
        assert isinstance(el, Array)
        assert np.allclose(el.data, a * b)


@pytest.mark.parametrize("a, b", TEST_VALUES)
def test_truediv(a, b):
    if isinstance(a, np.ndarray):
        a_trans = Array(a)
    else:
        a_trans = Float32(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Float32(b)


    res_1 = a / b_trans
    res_2 = a_trans / b
    res_3 = a_trans / b_trans

    for el in (
            res_1,
            res_2,
            res_3
    ):
        assert isinstance(el, Array)
        assert np.allclose(el.data, a / b)


@pytest.mark.parametrize("a, b", TEST_VALUES)
def test_pow(a, b):
    if isinstance(a, np.ndarray):
        a_trans = Array(a)
    else:
        a_trans = Float32(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Float32(b)

    res_1 = a ** b_trans
    res_2 = a_trans ** b
    res_3 = a_trans ** b_trans

    for el in (
            res_1,
            res_2,
            res_3
    ):
        assert isinstance(el, Array)
        assert np.allclose(
            el.data,
            a.astype(np.float32) ** b.astype(np.float32)
        )


@pytest.mark.parametrize(
    "dtype_a, dtype_b, result_dtype", TEST_DTYPES
)
def test_add_dtype(dtype_a, dtype_b, result_dtype):
    a = np.array([1, 2, 3], dtype=dtype_a)
    b = np.array([4, 5, 6], dtype=dtype_b)

    a_array = Array(a, dtype=dtype_a)
    b_array = Array(b, dtype=dtype_b)

    result_1 = a_array + b_array
    result_2 = a_array + b

    result_3 = a + b_array

    result_4 = a_array + 1
    result_5 = 2 + b_array

    for el in (
        result_1,
        result_2,
        result_3,
    ):
        assert el.data.dtype == result_dtype
        assert np.allclose(el.data, a + b)

    assert result_4.data.dtype == np.float64
    assert np.allclose(result_4.data, a + 1)
    assert result_5.data.dtype == np.float64
    assert np.allclose(result_5.data, b + 2)


@pytest.mark.parametrize(
    "dtype_a, dtype_b, result_dtype", TEST_DTYPES
)
def test_mul_dtype(dtype_a, dtype_b, result_dtype):
    a = np.array([1, 2, 3], dtype=dtype_a)
    b = np.array([4, 5, 6], dtype=dtype_b)

    a_array = Array(a, dtype=dtype_a)
    b_array = Array(b, dtype=dtype_b)

    result_1 = a_array * b_array
    result_2 = a_array * b

    result_3 = a * b_array

    result_4 = a_array * 2
    result_5 = 2 * b_array

    for el in (
        result_1,
        result_2,
        result_3,
    ):
        assert el.data.dtype == result_dtype
        assert np.allclose(el.data, a * b)

    assert result_4.data.dtype == np.float64
    assert np.allclose(result_4.data, a * 2)
    assert result_5.data.dtype == np.float64
    assert np.allclose(result_5.data, b * 2)


@pytest.mark.parametrize(
    "dtype_a, dtype_b, result_dtype", TEST_DTYPES
)
def test_truediv_dtype(dtype_a, dtype_b, result_dtype):
    a = np.array([1, 2, 3], dtype=dtype_a)
    b = np.array([4, 5, 6], dtype=dtype_b)

    a_array = Array(a, dtype=dtype_a)
    b_array = Array(b, dtype=dtype_b)

    result_1 = a_array / b_array
    result_2 = a_array / b

    result_3 = a / b_array

    result_4 = a_array / 2
    result_5 = 2 / b_array

    for el in (
        result_1,
        result_2,
        result_3,
    ):
        assert el.data.dtype == result_dtype
        assert np.allclose(el.data, a / b)

    assert result_4.data.dtype == np.float64
    assert np.allclose(result_4.data, a / 2)
    assert result_5.data.dtype == np.float64
    assert np.allclose(result_5.data, 2 / b)