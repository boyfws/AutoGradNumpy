import numpy as np
import numpy.typing as npt
from typing import Union, Any
from .add import add_backward
from src._types import Floatable, GradFnArray


def sub_backward(
        a: npt.NDArray,
        b: Union[npt.NDArray, Floatable],
        result: npt.NDArray
) -> GradFnArray:

    target_fn = add_backward(
        a, b, result
    )

    def fn(
            prev_grad: npt.NDArray[np.float32]
    ) -> tuple[
        npt.NDArray[np.float32],
        Union[npt.NDArray[np.float32], Floatable]
    ]:
        res = target_fn(prev_grad)
        return res[0], -res[1]

    return fn
