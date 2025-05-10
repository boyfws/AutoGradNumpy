from typing import Any, Callable, Union, cast

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, NumericDtypes

from .unbroadcast import unbroadcast


def pow_backward(
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

    mask_val = (a != 0) | np.zeros_like(result, dtype=np.bool_)
    mask_power = b == 1

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_a_temp = b * result / a  # type: ignore[operator]
        grad_a_temp = cast(npt.NDArray[Any], grad_a_temp)
        grad_a_raw = np.where(
            mask_val, grad_a_temp, np.where(mask_power, 1.0, 0.0)  # type: ignore[operator]
        ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_b_raw = np.where(mask_val, result * np.log(a), 0.0).astype(np.float32)

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[
        ArGradType,
        ArGradType,
    ]:
        grad_a = grad_a_raw * prev_grad
        grad_a = cast(ArGradType, grad_a)
        grad_a = unbroadcast(grad_a, a_shape)

        grad_b = grad_b_raw * prev_grad
        grad_b = cast(ArGradType, grad_b)
        grad_b = unbroadcast(grad_b, b_shape)

        return grad_a, grad_b

    return fn
