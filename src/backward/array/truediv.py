from typing import Any, Union

import numpy as np
import numpy.typing as npt

from src.types import Floatable, GradFnArray

from .unbroadcast import unbroadcast


def truediv_backward(
    a: Union[npt.NDArray[Any], Floatable],
    b: Union[npt.NDArray[Any], Floatable],
    result: npt.NDArray[Any],
) -> GradFnArray:

    a_shape = a.shape if isinstance(a, np.ndarray) else None

    b_shape = b.shape if isinstance(b, np.ndarray) else None

    def fn(
        prev_grad: npt.NDArray[np.float32],
    ) -> tuple[
        Union[npt.NDArray[np.float32], Floatable],
        Union[npt.NDArray[np.float32], Floatable],
    ]:
        grad_a_raw = prev_grad / b  # type: ignore[operator]
        grad_a = unbroadcast(grad_a_raw, a_shape)

        grad_b_raw = -prev_grad * result / b
        grad_b = unbroadcast(grad_b_raw, b_shape)

        return grad_a, grad_b

    return fn
