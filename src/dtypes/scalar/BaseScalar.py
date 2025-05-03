import numpy as np
from typing import Union, Callable, Optional
from src.backward import *


class BaseScalar:
    _dtype: np.float16 | np.float32 | np.float64
    prev_1: Optional["BaseScalar"]
    prev_2: Optional["BaseScalar"]
    grad: Optional[float]
    grad_fn: Optional[
        Callable[
            [float, float, float],
            tuple[
                Callable[[], float],
                Callable[[], float]
            ]
        ]
    ]

    def __init__(
            self,
            value: float,
            requires_grad: bool = False
    ) -> None:
        self.value = self._dtype(value)
        self.requires_grad = requires_grad

        self.prev_1 = None
        self.prev_2 = None
        self.grad = None
        self.grad_fn = None

    def __str__(self) -> str:
        return str(self.value)

    def item(self) -> float:
        return self.value

    def _backward(self, prev_grad: float) -> None:
        if self.requires_grad:
            if self.grad_fn is None:
                grad = 1
            else:
                grad = self.grad_fn()

            full_grad = prev_grad * grad

            if self.grad_fn is not None:
                if self.grad is None:
                    self.grad = full_grad
                else:
                    self.grad += full_grad

            if self.prev_1 is not None:
                self.prev_1._backward(full_grad)

            if self.prev_2 is not None:
                self.prev_2._backward(full_grad)

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad = 0

    def backward(self) -> None:
        self._backward(1)

    def detach(self) -> "BaseScalar":
        result_obj = object.__new__(type(self))
        result_obj.__init__(
            self.value,
            requires_grad=self.requires_grad
        )
        return result_obj

    def _base_operations_wrapper(
            self,
            other: Union["BaseScalar", float],
            fn_getter: Callable,
            operation_name: str
    ):
        flag = isinstance(other, BaseScalar)
        requires_grad = False
        if flag:
            sec = other.value
            requires_grad = other.requires_grad
        else:
            sec = other
        result = self.value.__getattribute__(operation_name)(sec)
        fn_1, fn_2 = fn_getter(self.value, sec, result)
        self.grad_fn = fn_1

        result_obj = object.__new__(type(self))
        result_obj.__init__(
            result,
            requires_grad=self.requires_grad or requires_grad
        )
        result_obj.prev_1 = self
        if flag:
            result_obj.prev_2 = other
            other.grad_fn = fn_2

        return result_obj

    def __neg__(self) -> "BaseScalar":
        result_obj = object.__new__(type(self))
        result_obj.__init__(
            -self.value,
            requires_grad=self.requires_grad
        )
        self.grad_fn = lambda: -1
        result_obj.prev_1 = self

        return result_obj

    def __add__(self,
                other: Union["BaseScalar", float]
                ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            add_backward,
            "__add__")

    def __radd__(self, other: float) -> "BaseScalar":
        return self.__add__(other)

    def __sub__(self, other: Union["BaseScalar", float]) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            sub_backward,
            "__sub__"
        )

    def __rsub__(self, other: float) -> "BaseScalar":
        return self.__neg__() + other

    def __mul__(self,
                other: Union["BaseScalar", float]
                ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            mul_backward,
            "__mul__"

        )

    def __rmul__(self, other: float) -> "BaseScalar":
        return self.__mul__(other)

    def __truediv__(self,
                    other: Union["BaseScalar", float]):
        return self._base_operations_wrapper(
            other,
            truediv_backward,
            "__truediv__"
        )

    def __rtruediv__(self, other) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            lambda a, b, c: truediv_backward(b, a, c)[::-1],
            "__rtruediv__"
        )

    def __pow__(
            self,
            power: Union["BaseScalar", float]
    ) -> "BaseScalar":
        return self._base_operations_wrapper(
            power,
            power_backward,
            "__pow__"
        )

    def __rpow__(self, other) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            lambda a, b, c: power_backward(b, a, c)[::-1],
            "__rpow__"
        )

    def __eq__(self, other: Union["BaseScalar", float]) -> bool:
        if isinstance(other, BaseScalar):
            return self.value == other.value
        else:
            return self.value == other



