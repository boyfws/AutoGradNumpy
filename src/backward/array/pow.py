from typing import Union, Any
from ._unbroadcast import _unbroadcast
import numpy as np
import numpy.typing as npt
from src._types import GradFnArray, Floatable


def pow_backward(
    a: Union[npt.NDArray, Floatable],
    b: Union[npt.NDArray, Floatable],
    result: npt.NDArray
) -> GradFnArray:

    a_shape = a.shape if isinstance(a, np.ndarray) else None
    b_shape = b.shape if isinstance(b, np.ndarray) else None

    def fn(
            prev_grad: npt.NDArray[np.float32]
    ) -> tuple[
        Union[npt.NDArray[np.float32], Floatable],
        Union[npt.NDArray[np.float32], Floatable]
    ]:

        grad_a_raw = b * prev_grad * result / (b + 1e-8)  # type: ignore[operator]
        grad_a = _unbroadcast(grad_a_raw, a_shape)

        grad_b_raw = prev_grad * result * np.log(a + 1e-8)  # type: ignore[operator]
        grad_b = _unbroadcast(grad_b_raw, b_shape)

        return grad_a, grad_b

    return fn
