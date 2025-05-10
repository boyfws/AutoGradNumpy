from typing import Callable, Union

import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, NumericDtypes

from .pow import pow_backward


def rpow_backward(
    a: ArrayValueType,
    b: Union[Floatable, npt.NDArray[NumericDtypes]],
    c: ArrayValueType,
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]:
    result = pow_backward(b, a, c)

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[
        ArGradType,
        ArGradType,
    ]:
        ret1, ret2 = result(prev_grad)
        return ret2, ret1

    return fn
