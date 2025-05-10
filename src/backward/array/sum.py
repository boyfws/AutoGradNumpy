from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable


def sum_backward(
    value: ArrayValueType,
    result: Union[npt.NDArray[np.float_], Floatable],
    axis: Optional[int] = None,
) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(prev_grad: ArGradType) -> tuple[ArGradType, None]:
        if axis is None:
            grad = np.broadcast_to(prev_grad, value.shape)
        else:
            grad = np.expand_dims(prev_grad, axis=axis)
            grad = np.broadcast_to(grad, value.shape)

        return grad.astype(np.float32), None

    return fn
