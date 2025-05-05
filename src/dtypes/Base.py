import numpy as np

import abc
from typing import Union, Callable, Optional, Type
from src.backward.scalar import *
from src.dtypes._EmptyCallable import _EmptyCallable


class BaseArray(abc.ABC):
    _ret_scalar_dtype: Type["BaseScalar"]
    _dtype: Union[np.float16, np.float32, np.float64]
    prev_1: Optional[
        Union["BaseScalar", "BaseArray"]
    ]
    prev_2: Optional[
        Union["BaseScalar", "BaseArray"]
    ]

    grad: Optional[np.ndarray]
    grad_fn: ...

    def item(self) -> np.ndarray:
        pass

    def _zero_grad(self) -> None:
        pass

    def _backward(
            self,
            prev_grad: np.ndarray
    ) -> None:
        pass

    def _graph_clean_up(self):
        pass

    def detach(self) -> "BaseArray":
        pass


class BaseScalar:
    _dtype: Union[
        np.float16,
        np.float32,
        np.float64
    ]
    prev_1: Optional[
        Union["BaseScalar", "BaseArray"]
    ]
    prev_2: Optional[
        Union["BaseScalar", "BaseArray"]
    ]
    grad: Optional[float]
    grad_fn: Optional[
        Callable[
            [],
            tuple[
                Union[float, np.ndarray, None],
                Union[float, np.ndarray, None],
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

    def _array_trigger(self,
                       other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ]
                       ) -> None:
        if isinstance(other, BaseArray):
            raise NotImplementedError

    def __str__(self) -> str:
        return str(self.value)

    def item(self) -> Union[
        np.float16,
        np.float32,
        np.float64
    ]:
        return self.value

    def _zero_grad(self) -> None:
        if self.grad is not None:
            self.grad = 0

    def _graph_clean_up(self) -> None:
        if self.grad_fn is not None:
            self.grad_fn = _EmptyCallable()

        if self.prev_1 is not None:
            self.prev_1._graph_clean_up()
            self.prev_1 = None

        if self.prev_2 is not None:
            self.prev_2._graph_clean_up()
            self.prev_2 = None

    def _backward(
            self,
            prev_grad: float,
    ) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = prev_grad
            else:
                self.grad += prev_grad

        if self.grad_fn is not None:

            if isinstance(self.grad_fn, _EmptyCallable):
                raise RuntimeError("The computational graph was cleaned up after the backward")

            grad1, grad2 = self.grad_fn()

            if grad1 is not None and self.prev_1 is not None:
                full_grad1 = grad1 * prev_grad
                self.prev_1._backward(full_grad1)

            if grad2 is not None and self.prev_2 is not None:
                full_grad2 = grad2 * prev_grad
                self.prev_2._backward(full_grad2)

    def backward(
            self,
            retain_graph: bool = False
    ) -> None:
        self._backward(1)

        if not retain_graph:
            self._graph_clean_up()

    def detach(self) -> "BaseScalar":
        result_obj = object.__new__(type(self))
        result_obj.__init__(
            self.value,
            requires_grad=False
        )
        return result_obj

    def _base_operations_wrapper(
            self,
            other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ],
            fn_getter: Callable,
            operation_name: str
    ) -> "BaseScalar":
        self._array_trigger(other)
        flag = isinstance(other, BaseScalar)
        value = self.item()

        if flag:
            sec = other.value
        else:
            sec = other

        result = value.__getattribute__(operation_name)(sec)
        fn = fn_getter(value, sec, result)

        result_obj = object.__new__(type(self))
        result_obj.__init__(
            result,
            requires_grad=False
        )
        result_obj.grad_fn = fn

        result_obj.prev_1 = self
        if flag:
            result_obj.prev_2 = other

        return result_obj

    def __neg__(self) -> "BaseScalar":
        result_obj = object.__new__(type(self))
        result_obj.__init__(
            -self.value,
            requires_grad=False
        )
        result_obj.grad_fn = neg_backward()
        result_obj.prev_1 = self

        return result_obj

    def __add__(self,
                other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ]
                ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            add_backward,
            "__add__")

    def __radd__(self, other: float) -> "BaseScalar":
        return self.__add__(other)

    def __sub__(self, other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ]
                ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            sub_backward,
            "__sub__"
        )

    def __rsub__(self, other: float) -> "BaseScalar":
        return self.__neg__() + other

    def __mul__(self,
                other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ]
                ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            mul_backward,
            "__mul__"

        )

    def __rmul__(self, other: float) -> "BaseScalar":
        return self.__mul__(other)

    def __truediv__(self,
                    other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ]
                    ):
        return self._base_operations_wrapper(
            other,
            truediv_backward,
            "__truediv__"
        )

    def __rtruediv__(self,
                     other: float
                     ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            lambda a, b, c: lambda: truediv_backward(b, a, c)()[::-1],
            "__rtruediv__"
        )

    def __pow__(
            self,
            other: Union[
                           float,
                           "BaseScalar",
                           "BaseArray"
                       ]
    ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            power_backward,
            "__pow__"
        )

    def __rpow__(self,
                 other: float
                 ) -> "BaseScalar":
        return self._base_operations_wrapper(
            other,
            lambda a, b, c: lambda: power_backward(b, a, c)()[::-1],
            "__rpow__"
        )

    def __eq__(self,
               other: Union["BaseScalar", float]
               ) -> bool:
        if isinstance(other, BaseScalar):
            return self.value == other.value
        else:
            return self.value == other
