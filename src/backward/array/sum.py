from typing import Any, Literal, Union, overload

import numpy as np
import numpy.typing as npt

from src.types import GradFnArray, GradFnScalar


@overload
def sum_backward(
    value: npt.NDArray[Any],
    axis: None,
) -> GradFnScalar: ...


@overload
def sum_backward(
    value: np.ndarray[tuple[int], Any],
    axis: Literal[0],
) -> GradFnScalar: ...


@overload
def sum_backward(
    value: npt.NDArray[Any],
    axis: int,
) -> GradFnArray: ...


def sum_backward(
    value: npt.NDArray[Any],
    axis: Union[int, None],
) -> Union[
    GradFnArray,
    GradFnScalar,
]:

    if axis is None or (axis == 0 and value.ndim == 1):

        def fn() -> tuple[npt.NDArray[np.float32], None]:
            return np.ones_like(value, dtype=np.float32), None

        return fn
    else:

        def fn(
            prev_grad: npt.NDArray[np.float32],
        ) -> tuple[npt.NDArray[np.float32], None]:
            grad = np.expand_dims(prev_grad, axis=axis)
            grad = np.broadcast_to(grad, value.shape)
            return grad.astype(np.float32), None

        return fn
