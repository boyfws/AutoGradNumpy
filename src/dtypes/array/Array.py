from typing import Any, Callable, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from src.backward.array import (
    add_backward,
    mul_backward,
    neg_backward,
    pow_backward,
    rpow_backward,
    rtruediv_backward,
    sub_backward,
    sum_backward,
    truediv_backward,
)
from src.dtypes.Base import BaseArray, BaseScalar
from src.dtypes.EmptyCallable import EmptyCallable
from src.dtypes.scalar import Float16, Float32, Float64
from src.types import Floatable


class Array(BaseArray):
    __array_priority__ = 1000

    def __init__(
        self,
        array: npt.ArrayLike,
        dtype: Optional[
            Union[Type[np.float16], Type[np.float32], Type[np.float64]]
        ] = None,
        requires_grad: bool = False,
    ):
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        if dtype is None:
            self._dtype = np.float32
            self._ret_scalar_dtype = Float32
        else:
            if dtype == np.float16:
                self._dtype = np.float16
                self._ret_scalar_dtype = Float16
            elif dtype == np.float32:
                self._dtype = np.float32
                self._ret_scalar_dtype = Float32
            elif dtype == np.float64:
                self._dtype = np.float64
                self._ret_scalar_dtype = Float64
            else:
                raise TypeError("dtype must be np.float16, np.float32, or np.float64")

        self.value = array.copy().astype(self._dtype)

        self.requires_grad = requires_grad
        self.prev_1 = None
        self.prev_2 = None
        self.grad = None
        self.grad_fn = None

    def __str__(self) -> str:
        return str(self.value)

    @property
    def data(self) -> npt.NDArray[Any]:
        return self.value.copy()

    def __getitem__(
        self, key: Any
    ) -> Union[npt.NDArray[Union[np.float16, np.float32, np.float64]], Floatable]:
        return self.value.__getitem__(key)

    def __setitem__(
        self, key: Any, value: Union[npt.ArrayLike[Any], Floatable]
    ) -> None:
        self.value.__setitem__(key, value)

        self.requires_grad = False

        if self.grad is not None:
            self.grad = np.zeros_like(self.value, dtype=np.float32)

    def _backward(self, prev_grad: npt.NDArray[np.float32]) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = prev_grad
            else:
                self.grad += prev_grad

        if self.grad_fn is not None:

            if isinstance(self.grad_fn, EmptyCallable):
                raise RuntimeError(
                    "The computational graph was cleaned up after the backward"
                )

            full_grad1, full_grad2 = self.grad_fn(prev_grad)

            if full_grad1 is not None and self.prev_1 is not None:
                self.prev_1._backward(full_grad1)

            if full_grad2 is not None and self.prev_2 is not None:
                self.prev_2._backward(full_grad2)

    def _zero_grad(self) -> None:
        if self.grad is not None:
            self.grad = np.zeros_like(self.value, dtype=self._dtype)

    def _graph_clean_up(self) -> None:
        if self.grad_fn is not None:
            self.grad_fn = EmptyCallable()

        if self.prev_1 is not None:
            self.prev_1._graph_clean_up()
            self.prev_1 = None

        if self.prev_2 is not None:
            self.prev_2._graph_clean_up()
            self.prev_2 = None

    def detach(self) -> "BaseArray":
        result_obj = type(self)(
            self.value,
            requires_grad=False,
            dtype=self._dtype,
        )
        return result_obj

    @staticmethod
    def _promote_type(
        a: npt.NDArray[Union[np.float16, np.float32, np.float64]],
        b: Union[npt.NDArray[Any], Floatable, np.float16, np.float32, np.float64],
    ) -> Union[Type[np.float16], Type[np.float32], Type[np.float64]]:
        if isinstance(b, np.ndarray):
            b_dtype = b.dtype
            if b_dtype not in (np.float16, np.float32, np.float64):
                b_dtype = np.float64
        elif isinstance(b, np.float16):
            b_dtype = np.float16
        elif isinstance(b, np.float32):
            b_dtype = np.float32
        elif isinstance(b, np.float64):
            b_dtype = np.float64
        else:
            b_dtype = np.float64

        result_dtype = np.promote_types(a.dtype, b_dtype).type

        return result_dtype

    def _base_operations_wrapper(
        self,
        other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"],
        fn_getter: Callable,
        operation_name: str,
    ) -> "BaseArray":
        scalar_flag = isinstance(other, BaseScalar)
        array_flag = isinstance(other, BaseArray)

        value = self.data
        if scalar_flag:
            sec = other.item()
        elif array_flag:
            sec = other.data
        else:
            sec = other

        result = value.__getattribute__(operation_name)(sec)

        result_dtype = self._promote_type(value, sec)
        result_obj = type(self)(
            result,
            requires_grad=False,
            dtype=result_dtype,
        )

        fn = fn_getter(self.value, sec, result)
        result_obj.grad_fn = fn

        result_obj.prev_1 = self
        if scalar_flag or array_flag:
            result_obj.prev_2 = other

        return result_obj

    def __neg__(self) -> "BaseArray":
        result_obj = type(self)(
            -self.value,
            requires_grad=False,
            dtype=self._dtype,
        )
        result_obj.grad_fn = neg_backward()
        result_obj.prev_1 = self

        return result_obj

    def __add__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(other, add_backward, "__add__")

    def __radd__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar"]
    ) -> "BaseArray":
        return self.__add__(other)

    def __sub__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(other, sub_backward, "__sub__")

    def __rsub__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar"]
    ) -> "BaseArray":
        return self.__neg__() + other  # type: ignore[operator]

    def __mul__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(other, mul_backward, "__mul__")

    def __rmul__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar"]
    ) -> "BaseArray":
        return self.__mul__(other)

    def __truediv__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(other, truediv_backward, "__truediv__")

    def __rtruediv__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(
            other,
            rtruediv_backward,
            "__rtruediv__",
        )

    def __pow__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(other, pow_backward, "__pow__")

    def __rpow__(
        self, other: Union[Floatable, npt.NDArray[Any], "BaseScalar"]
    ) -> "BaseArray":
        return self._base_operations_wrapper(
            other,
            rpow_backward,
            "__rpow__",
        )

    def __eq__(self, other: object) -> Union[bool, npt.NDArray[np.bool_]]:
        if isinstance(other, BaseArray):
            return self.data == other.data
        elif isinstance(other, np.ndarray):
            return self.data == other
        else:
            return False

    def sum(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]:
        value = self.data
        result = value.sum(axis=axis)
        if axis is None:
            result_obj = self._ret_scalar_dtype(
                result,
                requires_grad=False,
            )
        else:
            result_obj = type(self)(
                result,
                dtype=self._dtype,
                requires_grad=False,
            )

        result_obj.prev_1 = self

        grad = sum_backward(
            value,
            axis=axis,
        )
        result_obj.grad_fn = grad

        return result_obj

    def abs(self):
        pass
