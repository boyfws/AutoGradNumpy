from typing import Any, Callable, Union, cast

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
from src.types import Floatable, GradFnScalar, NotImplementedType


class Scalar(BaseScalar):
    __array_priority__ = 1000

    def __init__(self, value: Floatable, requires_grad: bool = False) -> None:
        converted_value = self._dtype(value)
        converted_value = cast(
            Union[np.float16, np.float32, np.float64], converted_value
        )
        self.value = converted_value
        self.requires_grad = requires_grad

        self.prev_1 = None
        self.prev_2 = None
        self.grad = None
        self.grad_fn = None

    @staticmethod
    def _array_trigger(
        other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"],
    ) -> Union[NotImplementedType, None]:
        if isinstance(other, BaseArray):
            return NotImplemented

        return None

    def __str__(self) -> str:
        return str(self.value)

    def item(self) -> Union[np.float16, np.float32, np.float64]:
        return self.value

    @property
    def data(self) -> Union[np.float16, np.float32, np.float64]:
        return self.item()

    @property
    def is_leaf(self) -> bool:
        return self.grad_fn is None

    def _zero_grad(self) -> None:
        if self.grad is not None:
            self.grad = 0

    def _graph_clean_up(self) -> None:
        if self.grad_fn is not None:
            self.grad_fn = EmptyCallable()

        if self.prev_1 is not None:
            self.prev_1._graph_clean_up()  # type: ignore[reportPrivateUsage]
            self.prev_1 = None

        if self.prev_2 is not None:
            self.prev_2._graph_clean_up()  # type: ignore[reportPrivateUsage]
            self.prev_2 = None

    def _backward(
        self,
        prev_grad: Floatable,
    ) -> None:
        if self.requires_grad and self.is_leaf:
            if self.grad is None:
                self.grad = prev_grad  # type: ignore[operator]
            else:
                self.grad += prev_grad  # type: ignore[operator]

        if self.grad_fn is not None and self.requires_grad:

            if isinstance(self.grad_fn, EmptyCallable):
                raise RuntimeError(
                    "The computational graph was cleaned up after the backward"
                )

            grad1, grad2 = self.grad_fn()

            if grad1 is not None and self.prev_1 is not None:
                full_grad1 = grad1 * prev_grad  # type: ignore[operator]
                self.prev_1._backward(full_grad1)  # type: ignore[reportPrivateUsage]

            if grad2 is not None and self.prev_2 is not None:
                full_grad2 = grad2 * prev_grad  # type: ignore[operator]
                self.prev_2._backward(full_grad2)  # type: ignore[reportPrivateUsage]

    @staticmethod
    def reverse_dunder(meth_name: str) -> str:
        if not meth_name.startswith("__") or not meth_name.endswith("__"):
            raise ValueError
        inner = meth_name[2:-2]
        if inner.startswith("r"):
            return f"__{inner[1:]}__"
        else:
            return f"__r{inner}__"

    @staticmethod
    def _convert_ndarray_to_base_array(array: npt.NDArray[Any]) -> "BaseArray":
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
        result_obj = type(self)(self.value, requires_grad=False)
        return result_obj

    def _base_operations_wrapper(
        self,
        other: Union["BaseScalar", Floatable, npt.NDArray[Any], "BaseArray"],
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> Union["BaseScalar", "BaseArray", NotImplementedType]:
        value = self.item()

        if isinstance(other, np.ndarray):
            array = self._convert_ndarray_to_base_array(other)
            return array.__getattribute__(self.reverse_dunder(operation_name))(value)

        block = self._array_trigger(other)
        if block is not None:
            return block

        flag = isinstance(other, BaseScalar)
        req_grad = self.requires_grad
        if flag:
            sec = other.item()
            req_grad = req_grad or other.requires_grad
        else:
            sec = other

        result = value.__getattribute__(operation_name)(sec)

        result_obj = type(self)(result, requires_grad=req_grad)

        if req_grad:
            fn = fn_getter(value, sec, result)

            result_obj.grad_fn = fn

            result_obj.prev_1 = self
            if flag:
                result_obj.prev_2 = other

        return result_obj

    def __neg__(self) -> "BaseScalar":
        result_obj = type(self)(-self.data, requires_grad=self.requires_grad)
        result_obj.grad_fn = neg_backward()
        result_obj.prev_1 = self

        return result_obj

    def __add__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, add_backward, "__add__")

    def __radd__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self.__add__(other)

    def __sub__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, sub_backward, "__sub__")

    def __rsub__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self.__neg__() + other  # type: ignore[operator]

    def __mul__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, mul_backward, "__mul__")

    def __rmul__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self.__mul__(other)

    def __truediv__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, truediv_backward, "__truediv__")

    def __rtruediv__(
        self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]
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
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]:
        return self._base_operations_wrapper(other, power_backward, "__pow__")

    def __rpow__(
        self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]
    ) -> Union[NotImplementedType, "BaseScalar", "BaseArray"]:
        return self._base_operations_wrapper(other, rpow_backward, "__rpow__")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BaseScalar):
            return self.data == other.data
        elif isinstance(other, Floatable):
            return self.data == other
        else:
            return False
