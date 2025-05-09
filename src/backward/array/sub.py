from typing import Callable, Union, overload, cast

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, GradFnArray, NumericDtypes

from .add import add_backward


@overload
def sub_backward(
    a: ArrayValueType, b: npt.NDArray[NumericDtypes], result: npt.NDArray[np.float_]
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]: ...


@overload
def sub_backward(
    a: ArrayValueType, b: Floatable, result: npt.NDArray[np.float_]
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        Floatable,
    ],
]: ...


def sub_backward(
    a: ArrayValueType,
    b: Union[npt.NDArray[NumericDtypes], Floatable],
    result: npt.NDArray[np.float_],
) -> GradFnArray:
    b_is_array = isinstance(b, np.ndarray)
    target_fn = add_backward(a, b, result)

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, Union[ArGradType, Floatable]]:
        res = target_fn(prev_grad)
        ret_1 = -res[1]  # type: ignore[operator]

        return res[0], ret_1

    if b_is_array:
        return cast(Callable[[ArGradType], tuple[ArGradType, ArGradType]], fn)
    else:
        return cast(Callable[[ArGradType], tuple[ArGradType, Floatable]], fn)