from typing import Callable, Union

import numpy as np

from src.types import ArGradType


def reshape_backward(
    original_shape: Union[list[int], tuple[int, ...]],
) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, None]:

        return np.reshape(prev_grad, newshape=original_shape), None

    return fn
