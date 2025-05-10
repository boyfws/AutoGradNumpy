from typing import Optional

import numpy as np

from src.types import ArGradType, ArrayValueType, GradFnArray


def prod_backward(
    value: ArrayValueType,
    result: ArrayValueType,
    axis: Optional[int] = None,
) -> GradFnArray:
    def fn(prev_grad: ArGradType) -> tuple[ArGradType, None]:
        if axis is None:
            res = result
            pg = prev_grad
        else:
            res = np.expand_dims(result, axis)
            pg = np.expand_dims(prev_grad, axis)

        res = np.broadcast_to(res, value.shape)
        pg = np.broadcast_to(pg, value.shape)

        grad = pg * (res / value)

        return grad.astype(np.float32), None

    return fn
