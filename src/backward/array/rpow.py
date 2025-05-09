from typing import Callable, Union, cast, overload

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, GradFnArray, NumericDtypes

from .pow import pow_backward


@overload
def rpow_backward(
    a: ArrayValueType, b: Floatable, c: npt.NDArray[np.float_]
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        Floatable,
    ],
]: ...


@overload
def rpow_backward(
    a: ArrayValueType, b: npt.NDArray[NumericDtypes], c: npt.NDArray[np.float_]
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]: ...


def rpow_backward(
    a: ArrayValueType,
    b: Union[Floatable, npt.NDArray[NumericDtypes]],
    c: npt.NDArray[np.float_],
) -> GradFnArray:
    b_is_array = isinstance(b, np.ndarray)
    result = pow_backward(b, a, c)

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[
        ArGradType,
        Union[ArGradType, Floatable],
    ]:
        ret1, ret2 = result(prev_grad)
        return ret2, ret1

    if b_is_array:
        return cast(Callable[[ArGradType], tuple[ArGradType, ArGradType]], fn)
    else:
        return cast(Callable[[ArGradType], tuple[ArGradType, Floatable]], fn)
