from typing import Callable, Any
from src._types import GradFnArray
import numpy as np
import numpy.typing as npt


def neg_backward() -> GradFnArray:

    def fn(
            prev_grad: npt.NDArray[np.float32]
    ) -> tuple[
        npt.NDArray[np.float32],
        None
    ]:
        return -prev_grad.copy(), None

    return fn