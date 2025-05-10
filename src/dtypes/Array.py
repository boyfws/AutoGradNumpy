from typing import Callable, Optional, Type, Union, cast

import numpy as np
import numpy.typing as npt

from src.backward.array import (
    abs_backward,
    add_backward,
    dot_backward,
    mul_backward,
    neg_backward,
    pow_backward,
    rpow_backward,
    rtruediv_backward,
    sub_backward,
    sum_backward,
    truediv_backward,
)
from src.dtypes.Base import BaseArray
from src.dtypes.EmptyCallable import EmptyCallable
from src.types import (
    ArGradType,
    ArrayValueType,
    BaseOperationsType,
    Floatable,
    NpIndicesTypes,
    NumericDtypes,
)


class Array(BaseArray):
    __array_priority__ = 1000

    def __init__(
        self,
        array: npt.ArrayLike,
        dtype: Optional[
            Union[Type[np.float16], Type[np.float32], Type[np.float64]]
        ] = None,
        requires_grad: bool = False,
    ) -> None:
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        if dtype is None:
            self._dtype = np.float32
        else:
            if dtype == np.float16:
                self._dtype = np.float16
            elif dtype == np.float32:
                self._dtype = np.float32
            elif dtype == np.float64:
                self._dtype = np.float64
            else:
                raise TypeError("dtype must be np.float16, np.float32, or np.float64")

        self._value = array.copy().astype(self._dtype)

        self._requires_grad = requires_grad
        self._prev_1 = None
        self._prev_2 = None
        self._grad = None
        self._grad_fn = None

    def __str__(self) -> str:
        return str(self._value)

    @property
    def data(self) -> ArrayValueType:
        return self._value.copy()

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape

    @property
    def size(self) -> int:
        return self._value.size

    @property
    def dtype(self) -> Union[Type[np.float16], Type[np.float32], Type[np.float64]]:
        return self._dtype

    @property
    def ndim(self) -> int:
        return self._value.ndim

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> ArGradType | None:
        if self._grad is not None:
            return self._grad.copy()

        return None

    def item(self):
        return self.data

    def __getitem__(
        self,
        key: Union[NpIndicesTypes, tuple[NpIndicesTypes, ...]],
    ) -> "BaseArray": ...

    def backward(self, retain_graph: bool = False) -> None:
        if self.shape == ():
            self._backward(np.array(1, dtype=np.float32))

        if not retain_graph:
            self._graph_clean_up()

    def _backward(self, prev_grad: ArGradType) -> None:
        if self.requires_grad and self.is_leaf:
            if self._grad is None:
                self._grad = prev_grad
            else:
                self._grad += prev_grad

        if self._grad_fn is not None and self.requires_grad:

            if isinstance(self._grad_fn, EmptyCallable):
                raise RuntimeError(
                    "The computational graph was cleaned up after the backward"
                )

            full_grad1, full_grad2 = self._grad_fn(prev_grad)

            self._prev_1._backward(full_grad1)  # type: ignore[reportPrivateUsage]

            if full_grad2 is not None and self._prev_2 is not None:
                self._prev_2._backward(full_grad2)  # type: ignore[reportPrivateUsage]

    def _zero_grad(self) -> None:
        if self._grad is not None:
            self._grad = np.zeros_like(self._value, dtype=np.float32)

    def _graph_clean_up(self) -> None:
        if self._grad_fn is not None:
            self._grad_fn = EmptyCallable()

        if self._prev_1 is not None:
            self._prev_1._graph_clean_up()  # type: ignore[reportPrivateUsage]
            self._prev_1 = None

        if self._prev_2 is not None:
            self._prev_2._graph_clean_up()  # type: ignore[reportPrivateUsage]
            self._prev_2 = None

    def detach(self) -> "BaseArray":
        result_obj = type(self)(
            self._value,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
        )
        return result_obj

    @staticmethod
    def _promote_type(
        a: ArrayValueType,
        b: Union[npt.NDArray[NumericDtypes], Floatable],
    ) -> Union[Type[np.float16], Type[np.float32], Type[np.float64]]:

        flag = isinstance(b, np.generic)

        if isinstance(b, np.ndarray):
            b_dtype = b.dtype
            if b_dtype not in (
                np.dtype(np.float16),
                np.dtype(np.float32),
                np.dtype(np.float64),
            ):
                b_dtype = np.dtype(np.float64)

        elif flag and (b.dtype == np.float16):
            b_dtype = np.float16
        elif flag and (b.dtype == np.float32):
            b_dtype = np.float32
        elif flag and (b.dtype == np.float64):
            b_dtype = np.float64
        else:
            b_dtype = np.float64

        result_dtype = np.promote_types(a.dtype, b_dtype).type

        return result_dtype

    def _base_operations_wrapper(
        self,
        other: BaseOperationsType,
        fn_getter: Callable[
            [
                ArrayValueType,
                Union[Floatable, npt.NDArray[NumericDtypes]],
                ArrayValueType,
            ],
            Callable[[ArGradType], tuple[ArGradType, ArGradType]],
        ],
        operation_name: str,
    ) -> "BaseArray":
        array_flag = isinstance(other, BaseArray)

        value = self.data
        req_grad = self.requires_grad

        if array_flag:
            sec = other.data
            req_grad = req_grad or other.requires_grad
        else:
            sec = other

        result = value.__getattribute__(operation_name)(sec)

        result_dtype = self._promote_type(value, sec)
        result_obj = type(self)(
            result,
            requires_grad=req_grad,
            dtype=result_dtype,
        )
        if req_grad:
            fn = fn_getter(value, sec, result_obj.data)
            result_obj._grad_fn = fn

            result_obj._prev_1 = self
            if array_flag:
                result_obj._prev_2 = other

        return result_obj

    def __neg__(self) -> "BaseArray":
        result_obj = type(self)(
            -self._value,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
        )
        if self.requires_grad:
            result_obj._grad_fn = neg_backward()
            result_obj._prev_1 = self

        return result_obj

    def __add__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(other, add_backward, "__add__")

    def __radd__(self, other: BaseOperationsType) -> "BaseArray":
        return self.__add__(other)

    def __sub__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(other, sub_backward, "__sub__")

    def __rsub__(self, other: BaseOperationsType) -> "BaseArray":
        return self.__neg__() + other  # type: ignore[operator]

    def __mul__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(other, mul_backward, "__mul__")

    def __rmul__(self, other: BaseOperationsType) -> "BaseArray":
        return self.__mul__(other)

    def __truediv__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(other, truediv_backward, "__truediv__")

    def __rtruediv__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(
            other,
            rtruediv_backward,
            "__rtruediv__",
        )

    def __pow__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(other, pow_backward, "__pow__")

    def __rpow__(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(
            other,
            rpow_backward,
            "__rpow__",
        )

    def __eq__(self, other: object) -> Union[np.bool_, npt.NDArray[np.bool_]]:  # type: ignore[reportIncompatibleMethodOverride]
        if isinstance(other, BaseArray):
            return cast(
                Union[np.bool_, npt.NDArray[np.bool_]], self._value == other.data
            )
        elif isinstance(other, (np.ndarray, int, float, np.floating)):
            return cast(Union[np.bool_, npt.NDArray[np.bool_]], self._value == other)
        else:
            return np.bool_(False)

    def __ne__(self, other: object) -> Union[np.bool_, npt.NDArray[np.bool_]]:  # type: ignore[reportIncompatibleMethodOverride]
        return ~(self == other)

    def __lt__(
        self, other: BaseOperationsType
    ) -> Union[np.bool_, npt.NDArray[np.bool_]]:
        pass

    def __le__(
        self, other: BaseOperationsType
    ) -> Union[np.bool_, npt.NDArray[np.bool_]]:
        pass

    def sum(self, axis: Optional[int] = None) -> "BaseArray":
        value = self.data
        result = value.sum(axis=axis)

        result_obj = type(self)(
            result,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:
            result_obj._prev_1 = self

            grad = sum_backward(
                value,
                axis=axis,
                result=result_obj.data,
            )
            result_obj._grad_fn = grad

        return result_obj

    def abs(self) -> "BaseArray":
        value = self.data
        result_obj = type(self)(
            np.abs(value),
            requires_grad=self.requires_grad,
            dtype=self.dtype,
        )
        if self.requires_grad:
            result_obj._grad_fn = abs_backward(value)
            result_obj._prev_1 = self

        return result_obj

    def dot(self, other: BaseOperationsType) -> "BaseArray":
        return self._base_operations_wrapper(
            other,
            dot_backward,
            "dot",
        )

    def mean(self, axis: Optional[int] = None) -> "BaseArray":
        if axis is None:
            n = self.size
        else:
            n = self.shape[axis]

        return self.sum(axis=axis) / n

    def transpose(self) -> "BaseArray":
        pass

    def reshape(self) -> "BaseArray":
        pass

    def min(self, axis: Optional[int] = None) -> "BaseArray":
        pass

    def max(self, axis: Optional[int] = None) -> "BaseArray":
        pass

    def prod(self, axis: Optional[int] = None) -> "BaseArray":
        pass

    def log(self) -> "BaseArray":
        pass

    def exp(self) -> "BaseArray":
        pass
