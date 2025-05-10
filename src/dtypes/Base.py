import abc
from typing import Callable, Literal, Optional, Type, Union, overload

import numpy as np
import numpy.typing as npt

from src.types import (
    ArGradType,
    ArrayValueType,
    BaseOperationsType,
    Floatable,
    GradFnArray,
    GradFnScalar,
    NotImplementedType,
    NpIndicesTypes,
    NumericDtypes,
)

from .EmptyCallable import EmptyCallable


class BaseArray(abc.ABC):
    _value: ArrayValueType
    _requires_grad: bool
    _dtype: Union[Type[np.float16], Type[np.float32], Type[np.float64]]

    _prev_1: Optional["BaseArray"]
    _prev_2: Optional[Union["BaseScalar", "BaseArray"]]

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

    @overload
    def get_item(
        self,
        key: Union[NpIndicesTypes, tuple[NpIndicesTypes, ...]],
        single_value: Literal[True],
    ) -> "BaseScalar": ...

    @overload
    def get_item(
        self,
        key: Union[NpIndicesTypes, tuple[NpIndicesTypes, ...]],
        single_value: Literal[False],
    ) -> "BaseArray": ...

    @abc.abstractmethod
    def get_item(
        self,
        key: Union[NpIndicesTypes, tuple[NpIndicesTypes, ...]],
        single_value: bool,
    ) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def _zero_grad(self) -> None: ...

    @abc.abstractmethod
    def _backward(self, prev_grad: ArGradType) -> None: ...

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

    @overload
    def _base_operations_wrapper(
        self,
        other: Union["BaseArray", npt.NDArray[NumericDtypes]],
        fn_getter: Callable[
            [
                ArrayValueType,
                npt.NDArray[NumericDtypes],
                npt.NDArray[np.float_],
            ],
            Callable[
                [ArGradType],
                tuple[
                    ArGradType,
                    ArGradType,
                ]
            ],
        ],
        operation_name: str,
    ) -> "BaseArray": ...

    @overload
    def _base_operations_wrapper(
        self,
        other: BaseOperationsType,
        fn_getter: Callable[
            [
                ArrayValueType,
                Union[Floatable, "BaseScalar"],
                npt.NDArray[np.float_],
            ],
            Callable[
                [ArGradType],
                tuple[
                    ArGradType,
                    Floatable,
                ]
            ],
        ],
        operation_name: str,
    ) -> "BaseArray": ...

    @abc.abstractmethod
    def _base_operations_wrapper(
        self,
        other: BaseOperationsType,
        fn_getter: Callable[
            [
                ArrayValueType,
                Union[Floatable, npt.NDArray[NumericDtypes]],
                npt.NDArray[np.float_],
            ],
            GradFnArray,
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
    def __lt__(self, other: BaseOperationsType) -> npt.NDArray[np.bool_]: ...

    @abc.abstractmethod
    def __le__(self, other: BaseOperationsType) -> npt.NDArray[np.bool_]: ...

    @overload
    def sum(self, axis: None = None) -> "BaseScalar": ...

    @overload
    def sum(self, axis: int) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def sum(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def abs(self) -> "BaseArray": ...

    @abc.abstractmethod
    def dot(
        self, other: Union["BaseArray", npt.NDArray[NumericDtypes]]
    ) -> "BaseArray": ...

    @overload
    def mean(self, axis: None = None) -> "BaseScalar": ...

    @overload
    def mean(self, axis: int) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def mean(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def transpose(self) -> "BaseArray": ...

    @abc.abstractmethod
    def reshape(self) -> "BaseArray": ...

    @overload
    def min(self, axis: None = None) -> "BaseScalar": ...

    @overload
    def min(self, axis: int) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def min(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]: ...

    @overload
    def max(self, axis: None = None) -> "BaseScalar": ...

    @overload
    def max(self, axis: int) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def max(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]: ...

    @overload
    def prod(self, axis: None = None) -> "BaseScalar": ...

    @overload
    def prod(self, axis: int) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def prod(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]: ...

    @abc.abstractmethod
    def log(self) -> "BaseArray": ...

    @abc.abstractmethod
    def exp(self) -> "BaseArray": ...


class BaseScalar(abc.ABC):
    _value: Union[np.float16, np.float32, np.float64]
    _dtype: Union[Type[np.float16], Type[np.float32], Type[np.float64]]
    _prev_1: Optional[Union["BaseScalar", "BaseArray"]]
    _prev_2: Optional[Union["BaseScalar", "BaseArray"]]
    _grad: Optional[Floatable]
    _grad_fn: Optional[Union[GradFnScalar, EmptyCallable]]
    _requires_grad: bool

    @abc.abstractmethod
    def item(self) -> Union[np.float16, np.float32, np.float64]: ...

    @property
    @abc.abstractmethod
    def data(self) -> Union[np.float16, np.float32, np.float64]: ...

    @property
    @abc.abstractmethod
    def is_leaf(self) -> bool: ...

    @property
    @abc.abstractmethod
    def grad(self) -> Union[Floatable, None]: ...

    @abc.abstractmethod
    def _zero_grad(self) -> None: ...

    @abc.abstractmethod
    def _graph_clean_up(self) -> None: ...

    @abc.abstractmethod
    def _backward(self, prev_grad: Floatable) -> None: ...

    @abc.abstractmethod
    def backward(self, retain_graph: bool = False) -> None: ...

    @abc.abstractmethod
    def detach(self) -> "BaseScalar": ...

    @staticmethod
    @overload
    def _array_trigger(other: "BaseArray") -> NotImplementedType: ...

    @staticmethod
    @overload
    def _array_trigger(
        other: Union[Floatable, npt.NDArray[NumericDtypes], "BaseScalar"],
    ) -> None: ...

    @staticmethod
    @abc.abstractmethod
    def _array_trigger(
        other: BaseOperationsType,
    ) -> Union[NotImplementedType, None]: ...

    @overload
    def _base_operations_wrapper(
        self,
        other: "BaseArray",
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> NotImplementedType: ...

    @overload
    def _base_operations_wrapper(
        self,
        other: npt.NDArray[NumericDtypes],
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> "BaseArray": ...

    @overload
    def _base_operations_wrapper(
        self,
        other: Union["BaseScalar", Floatable],
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> "BaseScalar": ...

    @abc.abstractmethod
    def _base_operations_wrapper(
        self,
        other: BaseOperationsType,
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> Union["BaseScalar", "BaseArray", NotImplementedType]: ...

    @overload
    def __add__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __add__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @overload
    def __add__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __add__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __radd__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __radd__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @overload
    def __radd__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @abc.abstractmethod
    def __radd__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __sub__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __sub__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @overload
    def __sub__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __sub__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rsub__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rsub__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @overload
    def __rsub__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rsub__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __mul__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __mul__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @overload
    def __mul__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __mul__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rmul__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rmul__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @overload
    def __rmul__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rmul__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __truediv__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __truediv__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @overload
    def __truediv__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __truediv__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rtruediv__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rtruediv__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @overload
    def __rtruediv__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rtruediv__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __pow__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __pow__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @overload
    def __pow__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __pow__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rpow__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rpow__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @overload
    def __rpow__(self, other: npt.NDArray[NumericDtypes]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rpow__(self, other: BaseOperationsType) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abc.abstractmethod
    def __ne__(self, other: object) -> bool: ...
