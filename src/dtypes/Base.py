import abc
from typing import Callable, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from src.types import (
    ArGradType,
    ArrayValueType,
    BaseOperationsType,
    Floatable,
    GradFnArray,
    NpIndicesTypes,
    NumericDtypes,
)

from .EmptyCallable import EmptyCallable


class BaseArray(abc.ABC):
    _value: ArrayValueType
    _requires_grad: bool
    _dtype: Union[Type[np.float16], Type[np.float32], Type[np.float64]]

    _prev_1: Optional["BaseArray"]
    _prev_2: Optional["BaseArray"]

    _grad: Optional[ArGradType]
    _grad_fn: Optional[Union[GradFnArray, EmptyCallable]]

    @abc.abstractmethod
    def __init__(
        self,
        array: npt.ArrayLike,
        dtype: Optional[
            Union[Type[np.float16], Type[np.float32], Type[np.float64]]
        ] = None,
        requires_grad: bool = False,
    ) -> None: ...

    @property
    @abc.abstractmethod
    def data(self) -> ArrayValueType: ...

    @property
    @abc.abstractmethod
    def is_leaf(self) -> bool: ...

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def dtype(self) -> Union[Type[np.float16], Type[np.float32], Type[np.float64]]: ...

    @property
    @abc.abstractmethod
    def ndim(self) -> int: ...

    @property
    @abc.abstractmethod
    def requires_grad(self) -> bool: ...

    @property
    @abc.abstractmethod
    def grad(self) -> Union[ArGradType, None]: ...

    @abc.abstractmethod
    def __getitem__(
        self,
        key: Union[NpIndicesTypes, tuple[NpIndicesTypes, ...]],
    ) -> "BaseArray": ...

    @abc.abstractmethod
    def _zero_grad(self) -> None: ...

    @abc.abstractmethod
    def _backward(self, prev_grad: ArGradType) -> None: ...

    @abc.abstractmethod
    def backward(self) -> None: ...

    @abc.abstractmethod
    def _graph_clean_up(self): ...

    @abc.abstractmethod
    def detach(self) -> "BaseArray": ...

    @staticmethod
    @abc.abstractmethod
    def _promote_type(
        a: ArrayValueType,
        b: Union[npt.NDArray[NumericDtypes], Floatable],
    ) -> Union[Type[np.float16], Type[np.float32], Type[np.float64]]: ...

    @abc.abstractmethod
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
    ) -> "BaseArray": ...

    @abc.abstractmethod
    def __neg__(self) -> "BaseArray": ...

    @abc.abstractmethod
    def __add__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __radd__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __sub__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __rsub__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __mul__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __rmul__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __truediv__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __rtruediv__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __pow__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __rpow__(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> Union[np.bool_, npt.NDArray[np.bool_]]: ...  # type: ignore[reportIncompatibleMethodOverride]

    @abc.abstractmethod
    def __ne__(self, other: object) -> Union[np.bool_, npt.NDArray[np.bool_]]: ...  # type: ignore[reportIncompatibleMethodOverride]

    @abc.abstractmethod
    def __lt__(
        self, other: BaseOperationsType
    ) -> Union[np.bool_, npt.NDArray[np.bool_]]: ...

    @abc.abstractmethod
    def __le__(
        self, other: BaseOperationsType
    ) -> Union[np.bool_, npt.NDArray[np.bool_]]: ...

    @abc.abstractmethod
    def sum(self, axis: Optional[int] = None) -> "BaseArray": ...

    @abc.abstractmethod
    def abs(self) -> "BaseArray": ...

    @abc.abstractmethod
    def dot(self, other: BaseOperationsType) -> "BaseArray": ...

    @abc.abstractmethod
    def mean(self, axis: Optional[int] = None) -> "BaseArray": ...

    @abc.abstractmethod
    def transpose(
        self,
        axes: Optional[Union[list[int], tuple[int, ...]]] = None,
        copy: bool = True,
    ) -> "BaseArray": ...

    @abc.abstractmethod
    def reshape(self, shape: Union[list[int], tuple[int, ...]]) -> "BaseArray": ...

    @abc.abstractmethod
    def min(self, axis: Optional[int] = None) -> "BaseArray": ...

    @abc.abstractmethod
    def max(self, axis: Optional[int] = None) -> "BaseArray": ...

    @abc.abstractmethod
    def prod(self, axis: Optional[int] = None) -> "BaseArray": ...

    @abc.abstractmethod
    def log(self) -> "BaseArray": ...

    @abc.abstractmethod
    def exp(self) -> "BaseArray": ...
