from typing import Union, Any
from ._unbroadcast import _unbroadcast
import numpy as np
import numpy.typing as npt
from src._types import GradFnArray, Floatable


def mul_backward(
    a: npt.NDArray,
    b: Union[npt.NDArray, Floatable],
    result: npt.NDArray
) -> GradFnArray:
    a_shape = a.shape
    b_is_array = isinstance(b, np.ndarray)
    b_shape = b.shape if b_is_array else None

    def fn(
            prev_grad: npt.NDArray[np.float32]
    ) -> tuple[
        npt.NDArray[np.float32],
        Union[npt.NDArray[np.float32], Floatable],
    ]:
        grad_a_raw = (prev_grad * b).astype(np.float32)  # type: ignore[operator]
        grad_a = _unbroadcast(grad_a_raw, a_shape)

        grad_b_raw = (prev_grad * a).astype(np.float32)  # type: ignore[operator]
        grad_b = _unbroadcast(grad_b_raw, b_shape)

        return grad_a, grad_b

    return fn
