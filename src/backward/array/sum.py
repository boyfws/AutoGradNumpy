from typing import Any, Literal, Union, overload

import numpy as np
import numpy.typing as npt

from src.types import ArrayValueType, GradFnArray, GradFnScalar


@overload
def sum_backward(
    value: ArrayValueType,
    axis: None,
) -> GradFnScalar: ...


@overload
def sum_backward(
    value: np.ndarray[tuple[int], Any],
    axis: Literal[0],
) -> GradFnScalar: ...


@overload
def sum_backward(
    value: ArrayValueType,
    axis: int,
) -> GradFnArray: ...


def sum_backward(
    value: ArrayValueType,
    axis: Union[int, None],
) -> Union[
    GradFnArray,
    GradFnScalar,
]:

    if axis is None or (axis == 0 and value.ndim == 1):

        def fn1() -> tuple[npt.NDArray[np.float32], None]:
            return np.ones_like(value, dtype=np.float32), None

        return fn1
    else:

        def fn2(
            prev_grad: npt.NDArray[np.float32],
        ) -> tuple[npt.NDArray[np.float32], None]:
            grad = np.expand_dims(prev_grad, axis=axis)
            grad = np.broadcast_to(grad, value.shape)
            return grad.astype(np.float32), None

        return fn2
