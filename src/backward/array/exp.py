from typing import Callable

import numpy as np

from src.types import ArGradType, ArrayValueType


def exp_backward(
    result: ArrayValueType,
) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, None]:

        return (prev_grad * result).astype(np.float32), None

    return fn
