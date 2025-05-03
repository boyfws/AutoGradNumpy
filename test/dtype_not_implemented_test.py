import pytest
from src import Float32, Array
import numpy as np

array = np.array([1, 2, 3], dtype=np.float16)


def test_add_with_array_raises():
    """Test that Float32 + Array raises NotImplementedError"""
    float_val = Float32(5.0)
    array_val = Array(array)

    with pytest.raises(NotImplementedError):
        _ = float_val + array_val


def test_mul_with_array_raises():
    """Test that Float32 * Array raises NotImplementedError"""
    float_val = Float32(2.0)
    array_val = Array(array)

    with pytest.raises(NotImplementedError):
        _ = float_val * array_val


def test_sub_with_array_raises():
    """Test that Float32 - Array raises NotImplementedError"""
    float_val = Float32(10.0)
    array_val = Array(array)

    with pytest.raises(NotImplementedError):
        _ = float_val - array_val


def test_truediv_with_array_raises():
    """Test that Float32 / Array raises NotImplementedError"""
    float_val = Float32(10.0)
    array_val = Array(array)

    with pytest.raises(NotImplementedError):
        _ = float_val / array_val


def test_pow_with_array_raises():
    """Test that Float32 / Array raises NotImplementedError"""
    float_val = Float32(10.0)
    array_val = Array(array)

    with pytest.raises(NotImplementedError):
        _ = float_val ** array_val
