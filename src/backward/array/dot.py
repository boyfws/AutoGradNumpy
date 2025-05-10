from typing import Callable, Union

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, NumericDtypes

from .mul import mul_backward


def dot_backward(
    a: ArrayValueType,
    b: Union[npt.NDArray[NumericDtypes], Floatable],
    c: ArrayValueType,
) -> Callable[[ArGradType], tuple[ArGradType, ArGradType]]:

    a_is_scalar = a.shape == ()
    b_is_array = isinstance(b, np.ndarray)

    if a_is_scalar or not b_is_array or (b_is_array and b.shape == ()):
        return mul_backward(a, b, c)

    def fn(prev_grad: ArGradType) -> tuple[ArGradType, ArGradType]:
        return prev_grad.dot(b.T).astype(np.float32), a.T.dot(prev_grad).astype(
            np.float32
        )

    return fn
