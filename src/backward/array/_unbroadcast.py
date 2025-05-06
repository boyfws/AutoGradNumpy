from typing import (Any,
                    Union,
                    overload,
                    Literal,
                    Optional)
from src._types import Floatable

import numpy as np
import numpy.typing as npt


@overload
def _unbroadcast(
    grad: npt.NDArray[np.float32],
    shape: Literal[None]
) -> Floatable: ...


@overload
def _unbroadcast(
    grad: npt.NDArray[np.float32],
    shape: tuple[int, ...]
) -> npt.NDArray[np.float32]: ...


def _unbroadcast(
        grad: npt.NDArray[np.float32],
        shape: Optional[tuple[int, ...]] = None
) -> Union[
    npt.NDArray[np.float32],
    Floatable
]:
    """
    Reduce `grad` back to `shape` by summing over broadcasted dimensions.
    """
    if shape is None:
        return grad.sum().reshape(())

    grad_shape = grad.shape
    ndim_diff = grad.ndim - len(shape)
    shape_padded = (1,) * ndim_diff + shape  # e.g. (1,1,3,4)

    axes = tuple(i for i, dim in enumerate(shape_padded) if dim == 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)