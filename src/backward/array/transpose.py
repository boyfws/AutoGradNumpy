from typing import Callable, Union

import numpy as np

from src.types import ArGradType


def transpose_backward(
    axes: Union[list[int], tuple[int, ...], None],
) -> Callable[[ArGradType], tuple[ArGradType, None]]:

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[ArGradType, None]:

        return np.transpose(prev_grad, axes=axes), None

    return fn
