from typing import Callable

import numpy as np

from src.types import ArGradType, ArrayValueType


def abs_backward(
    array: ArrayValueType,
) -> Callable[[ArGradType], tuple[ArGradType, None]]:
    def fn(prev_grad: ArGradType) -> tuple[ArGradType, None]:
        return (np.sign(array) * prev_grad).astype(np.float32), None

    return fn
