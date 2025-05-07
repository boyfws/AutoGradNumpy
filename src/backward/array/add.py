from typing import Any, Union

import numpy as np
import numpy.typing as npt

from src.types import ArrayValueType, Floatable, GradFnArray

from .unbroadcast import unbroadcast


def add_backward(
    a: ArrayValueType, b: Union[npt.NDArray[Any], Floatable], result: npt.NDArray[Any]
) -> GradFnArray:

    a_shape = a.shape

    b_is_array = isinstance(b, np.ndarray)
    b_shape = b.shape if b_is_array else None

    def fn(
        prev_grad: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], Union[npt.NDArray[np.float32], Floatable]]:

        grad_a = unbroadcast(prev_grad, a_shape)
        grad_b = unbroadcast(prev_grad, b_shape)

        return grad_a, grad_b

    return fn
