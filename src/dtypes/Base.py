import abc
from typing import Any, Callable, Optional, Type, Union, overload

import numpy as np
import numpy.typing as npt

from src.types import Floatable, GradFnArray, GradFnScalar, NotImplementedType


class BaseArray(abc.ABC):
    _ret_scalar_dtype: Type["BaseScalar"]
    _dtype: Union[Type[np.float16], Type[np.float32], Type[np.float64]]
    prev_1: Optional[Union["BaseScalar", "BaseArray"]]
    prev_2: Optional[Union["BaseScalar", "BaseArray"]]

    value: npt.NDArray[Union[np.float16, np.float32, np.float64]]
    grad: Optional[npt.NDArray[np.float32]]
    grad_fn: Optional[GradFnArray]

    @abc.abstractmethod
    def __init__(
        self,
        array: npt.ArrayLike,
        dtype: Optional[Union[np.float16, np.float32, np.float64]] = None,
        requires_grad: bool = False,
    ) -> None: ...

    @abc.abstractmethod
    def _zero_grad(self) -> None: ...

    @abc.abstractmethod
    def _backward(self, prev_grad: npt.NDArray[Any]) -> None: ...

    @abc.abstractmethod
    def _graph_clean_up(self): ...

    @abc.abstractmethod
    def detach(self) -> "BaseArray": ...

    @property
    @abc.abstractmethod
    def data(self) -> npt.NDArray[Any]: ...

    @overload
    def sum(self, axis: None = None) -> "BaseScalar": ...

    @overload
    def sum(self, axis: int) -> "BaseArray": ...

    @abc.abstractmethod
    def sum(self, axis: Optional[int] = None) -> Union["BaseScalar", "BaseArray"]: ...


class BaseScalar(abc.ABC):
    _dtype: Union[Type[np.float16], Type[np.float32], Type[np.float64]]
    prev_1: Optional[Union["BaseScalar", "BaseArray"]]
    prev_2: Optional[Union["BaseScalar", "BaseArray"]]
    grad: Optional[Floatable]
    grad_fn: Optional[GradFnScalar]

    @abc.abstractmethod
    def item(self) -> Union[np.float16, np.float32, np.float64]: ...

    @property
    @abc.abstractmethod
    def data(self) -> Union[np.float16, np.float32, np.float64]: ...

    @abc.abstractmethod
    def _zero_grad(self) -> None: ...

    @abc.abstractmethod
    def _graph_clean_up(self) -> None: ...

    @abc.abstractmethod
    def _backward(self, prev_grad: Floatable) -> None: ...

    @abc.abstractmethod
    def backward(self, retain_graph: Optional[bool] = False) -> None: ...

    @abc.abstractmethod
    def detach(self) -> "BaseScalar": ...

    @staticmethod
    @overload
    def _array_trigger(other: BaseArray) -> NotImplementedType: ...

    @staticmethod
    @overload
    def _array_trigger(
        other: Union[Floatable, npt.NDArray[Any], "BaseScalar"],
    ) -> None: ...

    @staticmethod
    @abc.abstractmethod
    def _array_trigger(
        other: Union[Floatable, npt.NDArray[Any], "BaseScalar", "BaseArray"],
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
        other: npt.NDArray[Any],
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
        other: Union["BaseScalar", Floatable, npt.NDArray[Any], "BaseArray"],
        fn_getter: Callable[[Floatable, Floatable, Floatable], GradFnScalar],
        operation_name: str,
    ) -> Union["BaseScalar", "BaseArray", NotImplementedType]: ...

    @overload
    def __add__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __add__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @overload
    def __add__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __add__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __radd__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __radd__(self, other: Floatable) -> "BaseScalar": ...

    @overload
    def __radd__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @abc.abstractmethod
    def __radd__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __sub__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __sub__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @overload
    def __sub__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __sub__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rsub__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rsub__(self, other: Floatable) -> "BaseScalar": ...

    @overload
    def __rsub__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rsub__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __mul__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __mul__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @overload
    def __mul__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __mul__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rmul__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rmul__(self, other: Floatable) -> "BaseScalar": ...

    @overload
    def __rmul__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rmul__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __truediv__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __truediv__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @overload
    def __truediv__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __truediv__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rtruediv__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rtruediv__(self, other: Floatable) -> "BaseScalar": ...

    @overload
    def __rtruediv__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rtruediv__(
        self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __pow__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __pow__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @overload
    def __pow__(self, other: Union[Floatable, "BaseScalar"]) -> "BaseScalar": ...

    @abc.abstractmethod
    def __pow__(
        self, other: Union["BaseArray", "BaseScalar", Floatable, npt.NDArray[Any]]
    ) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @overload
    def __rpow__(self, other: "BaseArray") -> NotImplementedType: ...

    @overload
    def __rpow__(self, other: Floatable) -> "BaseScalar": ...

    @overload
    def __rpow__(self, other: npt.NDArray[Any]) -> "BaseArray": ...

    @abc.abstractmethod
    def __rpow__(self, other: Union[Floatable, "BaseArray", npt.NDArray[Any]]) -> Union[
        NotImplementedType,
        "BaseArray",
        "BaseScalar",
    ]: ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...
