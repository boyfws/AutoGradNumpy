from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt

from src.types import Floatable, GradFnArray

from .unbroadcast import unbroadcast


def pow_backward(
    a: Union[npt.NDArray[Any], Floatable],
    b: Union[npt.NDArray[Any], Floatable],
    result: npt.NDArray[Any],
) -> GradFnArray:

    a_shape = a.shape if isinstance(a, np.ndarray) else None
    b_shape = b.shape if isinstance(b, np.ndarray) else None

    mask_val = (a != 0) + np.zeros_like(result, dtype=np.bool_)
    mask_power = b == 1

    grad_a_raw = np.zeros_like(result, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_a_raw = np.where(
            mask_val, b * result / a, np.where(mask_power, 1.0, 0.0)
        ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_b_raw = np.where(mask_val, result * np.log(a), 0.0).astype(np.float32)

    def fn(
        prev_grad: npt.NDArray[np.float32],
    ) -> tuple[
        Union[npt.NDArray[np.float32], Floatable],
        Union[npt.NDArray[np.float32], Floatable],
    ]:
        grad_a = grad_a_raw * prev_grad
        grad_a = cast(npt.NDArray[np.float32], grad_a)
        grad_a = unbroadcast(grad_a, a_shape)

        grad_b = grad_b_raw * prev_grad
        grad_b = cast(npt.NDArray[np.float32], grad_b)
        grad_b = unbroadcast(grad_b, b_shape)

        return grad_a, grad_b

    return fn
