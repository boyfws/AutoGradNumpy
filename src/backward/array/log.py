from typing import Callable

import numpy as np

from src.types import ArGradType, ArrayValueType


def log_backward(a: ArrayValueType) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, None]:

        return (prev_grad / a).astype(np.float32), None

    return fn
