from typing import Any, Union, overload

import numpy as np
import numpy.typing as npt

from src.types import ArrayValueType, Floatable, GradFnArray, GradFnScalar, Optional


@overload
def sum_backward(
    value: ArrayValueType,
    result: Floatable,
    axis: Optional[int] = None,
) -> GradFnScalar: ...


@overload
def sum_backward(
    value: ArrayValueType,
    result: npt.NDArray[Any],
    axis: Optional[int] = None,
) -> GradFnArray: ...


def sum_backward(
    value: ArrayValueType,
    result: Union[npt.NDArray[Any], Floatable],
    axis: Optional[int] = None,
) -> Union[
    GradFnArray,
    GradFnScalar,
]:

    if not isinstance(result, np.ndarray):

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
