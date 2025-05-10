import numpy as np
import pytest

from add_src_to_path import append_src
append_src()

from src import Array

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


def test_init_not_array() -> None:
    array = Array([1, 2, 3], dtype=np.float32)
    assert all(array.data == np.array([1, 2, 3], dtype=np.float32))


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
        a_trans = Array(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Array(b)

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
        a_trans = Array(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Array(b)

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
        a_trans = Array(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Array(b)

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
        a_trans = Array(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Array(b)


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
        a_trans = Array(a)

    if isinstance(b, np.ndarray):
        b_trans = Array(b)
    else:
        b_trans = Array(b)

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


@pytest.mark.parametrize(
    "array", [
        np.array([1, 2, 3]),
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
    ]
)
def test_str(array):
    array = array.astype(np.float32)
    ar = Array(array)
    assert str(ar) == str(array)


def test_empty_zero_grad():
    a = Array(np.array([1, 2, 3]), requires_grad=True)
    a._zero_grad()
    assert a._grad is None


def test_not_empty_zero_grad():
    a = Array(np.array([1, 2, 3]), requires_grad=True)
    a.sum().backward()
    a._zero_grad()

    assert a._grad is not None
    assert np.allclose(np.zeros_like(a.data), a._grad)


def test_eq():
    a = Array(np.array([1, 2, 3]), requires_grad=False)
    b = Array(np.array([1, 2, 3]), requires_grad=False)
    c = Array(np.array([2, 2, 3]), requires_grad=False)

    assert all(a == a)
    assert all(c == c)
    assert all(a == b)
    assert any(c != b)
    assert any(a != 2)
