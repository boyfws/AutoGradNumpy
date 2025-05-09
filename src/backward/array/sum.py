from typing import Callable, Optional, Union, cast, overload

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, GradFnArray, GradFnScalar


@overload
def sum_backward(
    value: ArrayValueType,
    result: Floatable,
    axis: Optional[int] = None,
) -> Callable[[], tuple[ArGradType, None]]: ...


@overload
def sum_backward(
    value: ArrayValueType,
    result: npt.NDArray[np.float_],
    axis: Optional[int] = None,
) -> Callable[[ArGradType], tuple[ArGradType, None]]: ...


def sum_backward(
    value: ArrayValueType,
    result: Union[npt.NDArray[np.float_], Floatable],
    axis: Optional[int] = None,
) -> Union[
    GradFnArray,
    GradFnScalar,
]:

    if not isinstance(result, np.ndarray):

        def fn1() -> tuple[ArGradType, None]:
            result = np.ones_like(value, dtype=np.float32)

            return result, None

        return fn1
    else:
        axis = cast(int, axis)

        def fn2(
            prev_grad: ArGradType,
        ) -> tuple[ArGradType, None]:
            grad = np.expand_dims(prev_grad, axis=axis)
            grad = np.broadcast_to(grad, value.shape)
            return grad.astype(np.float32), None

        return fn2
