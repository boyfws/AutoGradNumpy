from typing import Union, Any
import numpy as np
import numpy.typing as npt
from ._unbroadcast import _unbroadcast
from src._types import GradFnArray, Floatable


def add_backward(
        a: npt.NDArray,
        b: Union[npt.NDArray, Floatable],
        result: npt.NDArray
) -> GradFnArray:

    a_shape = a.shape

    b_is_array = isinstance(b, np.ndarray)
    b_shape = b.shape if b_is_array else None

    def fn(prev_grad: npt.NDArray[np.float32]) -> tuple[
        npt.NDArray[np.float32],
        Union[
            npt.NDArray[np.float32],
            Floatable
        ]
    ]:

        grad_a = _unbroadcast(prev_grad, a_shape)
        grad_b = _unbroadcast(prev_grad, b_shape)

        return grad_a, grad_b

    return fn

