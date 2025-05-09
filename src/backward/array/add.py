from typing import Callable, Union, cast, overload

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, GradFnArray, NumericDtypes

from .unbroadcast import unbroadcast


@overload
def add_backward(
    a: ArrayValueType, b: npt.NDArray[NumericDtypes], result: npt.NDArray[np.float_]
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]: ...


@overload
def add_backward(
    a: ArrayValueType, b: Floatable, result: npt.NDArray[np.float_]
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        Floatable,
    ],
]: ...


def add_backward(
    a: ArrayValueType,
    b: Union[npt.NDArray[NumericDtypes], Floatable],
    result: npt.NDArray[np.float_],
) -> GradFnArray:

    a_shape = a.shape

    b_is_array = isinstance(b, np.ndarray)
    b_shape = b.shape if b_is_array else None

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, Union[ArGradType, Floatable]]:

        grad_a = unbroadcast(prev_grad, a_shape)
        grad_b = unbroadcast(prev_grad, b_shape)

        return grad_a, grad_b

    if b_is_array:
        return cast(Callable[[ArGradType], tuple[ArGradType, ArGradType]], fn)
    else:
        return cast(Callable[[ArGradType], tuple[ArGradType, Floatable]], fn)
