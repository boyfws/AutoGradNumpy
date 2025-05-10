from typing import Callable, Union

import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, NumericDtypes

from .add import add_backward


def sub_backward(
    a: ArrayValueType,
    b: Union[npt.NDArray[NumericDtypes], Floatable],
    result: ArrayValueType,
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]:
    target_fn = add_backward(a, b, result)

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, ArGradType]:
        res = target_fn(prev_grad)
        ret_1 = -res[1]  # type: ignore[operator]

        return res[0], ret_1

    return fn
