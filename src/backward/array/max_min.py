from typing import Callable, Union, Optional
from numpy.typing import NDArray
import numpy as np

from src.types import ArGradType


def max_min_backward(
    mask: NDArray[np.float32],
    axis: Optional[int] = None,
) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, None]:
        if axis is None:
            return prev_grad * mask, None
        else:
            idx = [1] * mask.ndim
            idx[axis] = -1

            grad = prev_grad.reshape(*idx) * mask
            return grad, None

    return fn
