from typing import Optional

import numpy as np

from src.types import ArGradType


def unbroadcast(
    grad: ArGradType, shape: Optional[tuple[int, ...]] = None
) -> ArGradType:
    """
    Reduce `grad` back to `shape` by summing over broadcasted dimensions.
    """
    if shape is None:
        return np.array(grad.sum(), dtype=np.float32)

    ndim_diff = grad.ndim - len(shape)
    shape_padded = (1,) * ndim_diff + shape  # e.g. (1,1,3,4)

    axes = tuple(i for i, dim in enumerate(shape_padded) if dim == 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)
