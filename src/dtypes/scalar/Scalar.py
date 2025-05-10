from typing import Callable, Union, cast

import numpy as np
import numpy.typing as npt

from src.backward.scalar import (
    add_backward,
    mul_backward,
    neg_backward,
    power_backward,
    rpow_backward,
    rtruediv_backward,
    sub_backward,
    truediv_backward,
)
from src.dtypes.Base import BaseArray, BaseScalar
from src.dtypes.EmptyCallable import EmptyCallable
from src.types import Floatable, GradFnScalar, NotImplementedType, BaseOperationsType, NumericDtypes


class Scalar(BaseScalar):
    __array_priority__ = 1000

    def __init__(self, value: Floatable, requires_grad: bool = False) -> None:
        converted_value = self._dtype(value)
        self._value = converted_value
        self._requires_grad = requires_grad

        self._prev_1 = None
        self._prev_2 = None
        self._grad = None
        self._grad_fn = None

    @staticmethod
    def _array_trigger(
        other: BaseOperationsType,
    ) -> Union[NotImplementedType, None]:
        if isinstance(other, BaseArray):
            return NotImplemented

        return None

    def __str__(self) -> str:
        return str(self._value)

    def item(self) -> Union[np.float16, np.float32, np.float64]:
        return self._value

    @property
    def data(self) -> Union[np.float16, np.float32, np.float64]:
        return self.item()

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    @property
    def grad(self) -> Union[Floatable, None]:
        return self._grad

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def _zero_grad(self) -> None:
        if self._grad is not None:
            self._grad = 0

    def _graph_clean_up(self) -> None:
        if self._grad_fn is not None:
            self._grad_fn = EmptyCallable()

        if self._prev_1 is not None:
            self._prev_1._graph_clean_up()  # type: ignore[reportPrivateUsage]
            self._prev_1 = None

        if self._prev_2 is not None:
            self._prev_2._graph_clean_up()  # type: ignore[reportPrivateUsage]
            self._prev_2 = None

    def _backward(
        self,
        prev_grad: Floatable,
    ) -> None:
        if self._requires_grad and self.is_leaf:
            if self._grad is None:
                self._grad = prev_grad  # type: ignore[operator]
            else:
                self._grad += prev_grad  # type: ignore[operator]

        if self._grad_fn is not None and self._requires_grad:

            if isinstance(self._grad_fn, EmptyCallable):
                raise RuntimeError(
                    "The computational graph was cleaned up after the backward"
                )

            grad1, grad2 = self._grad_fn()

            full_grad1 = grad1 * prev_grad  # type: ignore[operator]
            self._prev_1._backward(full_grad1)  # type: ignore[reportPrivateUsage]

            if grad2 is not None and self._prev_2 is not None:
                full_grad2 = grad2 * prev_grad  # type: ignore[operator]
                self._prev_2._backward(full_grad2)  # type: ignore[reportPrivateUsage]

    @staticmethod
    def _reverse_dunder(meth_name: str) -> str:
        if not meth_name.startswith("__") or not meth_name.endswith("__"):
            raise ValueError
        inner = meth_name[2:-2]
        if inner.startswith("r"):
            return f"__{inner[1:]}__"
        else:
            return f"__r{inner}__"

    @staticmethod
    def _convert_ndarray_to_base_array(array: npt.NDArray[NumericDtypes]) -> "BaseArray":
        from src.dtypes import Array

        if array.dtype == np.float16:
            dtype = np.float16
        elif array.dtype == np.float32:
            dtype = np.float32
        elif array.dtype == np.float64:
            dtype = np.float64
        else:
            dtype = np.float32

        array_obj = Array(array, dtype=dtype, requires_grad=False)

        return array_obj

    def backward(self, retain_graph: bool = False) -> None:
        self._backward(1)

        if not retain_graph:
            self._graph_clean_up()

    def detach(self) -> "BaseScalar":
        result_obj = type(self)(self._value, requires_grad=False)
        return result_obj

    def _base_operations_wrapper(
        self,
        other: BaseOperationsType,
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> Union["BaseScalar", "BaseArray", NotImplementedType]:
        value = self.item()

        if isinstance(other, np.ndarray):
            array = self._convert_ndarray_to_base_array(other)
            return array.__getattribute__(self._reverse_dunder(operation_name))(value)
        # => Not ndarray

        block = self._array_trigger(other)
        if block is not None:
            return block
        # => not BaseArray

        other = cast(Union[BaseScalar, Floatable], other)

        flag = isinstance(other, BaseScalar)
        req_grad = self._requires_grad
        if flag:
            sec = other.item()
            req_grad = req_grad or other._requires_grad
        else:
            sec = other

        result = value.__getattribute__(operation_name)(sec)

        result_obj = type(self)(result, requires_grad=req_grad)

        if req_grad:
            fn = fn_getter(value, sec, result)

            result_obj._grad_fn = fn

            result_obj._prev_1 = self
            if flag:
                result_obj._prev_2 = other

        return result_obj

    def __neg__(self) -> "BaseScalar":
        result_obj = type(self)(-self.data, requires_grad=self._requires_grad)
        if self._requires_grad:
            result_obj._grad_fn = neg_backward()
            result_obj._prev_1 = self

        return result_obj

    def __add__(
        self, other: BaseOperationsType
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, add_backward, "__add__")

    def __radd__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self.__add__(other)

    def __sub__(
        self, other: BaseOperationsType
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, sub_backward, "__sub__")

    def __rsub__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self.__neg__() + other  # type: ignore[operator]

    def __mul__(
        self, other: BaseOperationsType
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, mul_backward, "__mul__")

    def __rmul__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self.__mul__(other)

    def __truediv__(
        self, other: BaseOperationsType
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, truediv_backward, "__truediv__")

    def __rtruediv__(
        self, other: BaseOperationsType
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(
            other,
            rtruediv_backward,
            "__rtruediv__",
        )

    def __pow__(
        self, other: BaseOperationsType
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, power_backward, "__pow__")

    def __rpow__(
        self, other: BaseOperationsType
    ) -> Union[NotImplementedType, "BaseScalar", "BaseArray"]:
        return self._base_operations_wrapper(other, rpow_backward, "__rpow__")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseScalar):
            return self.data == other.data
        elif isinstance(other, (int, float, np.floating)):
            return self.data == other
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not (self == other)
