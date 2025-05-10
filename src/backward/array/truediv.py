from collections.abc import Callable
from typing import Union

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, NumericDtypes

from .unbroadcast import unbroadcast


def truediv_backward(
    a: Union[npt.NDArray[NumericDtypes], Floatable],
    b: Union[npt.NDArray[NumericDtypes], Floatable],
    result: ArrayValueType,
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]:
    if isinstance(a, np.ndarray):
        a_shape = a.shape if a.shape != () else None
    else:
        a_shape = None

    if isinstance(b, np.ndarray):
        b_shape = b.shape if b.shape != () else None
    else:
        b_shape = None

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[
        ArGradType,
        ArGradType,
    ]:
        grad_a_raw = (prev_grad / b).astype(np.float32)  # type: ignore[operator]
        grad_a = unbroadcast(grad_a_raw, a_shape)

        grad_b_raw = (-prev_grad * result / b).astype(np.float32)
        grad_b = unbroadcast(grad_b_raw, b_shape)

        return grad_a, grad_b

    return fn
