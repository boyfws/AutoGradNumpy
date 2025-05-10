from typing import Callable, Union

import numpy as np

from src.types import ArGradType, ArrayValueType, NpIndicesTypes


def getitem_backward(
    a: ArrayValueType,
    index: Union[NpIndicesTypes, tuple[NpIndicesTypes, ...]],
) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, None]:

        real_grad = np.zeros_like(a, dtype=np.float32)
        real_grad[index] = prev_grad

        return real_grad, None

    return fn
