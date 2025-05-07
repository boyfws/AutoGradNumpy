from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt

from src.types import ArrayValueType, Floatable, GradFnArray

from .truediv import truediv_backward


def rtruediv_backward(
    a: ArrayValueType, b: Union[Floatable, npt.NDArray[Any]], c: npt.NDArray[Any]
) -> GradFnArray:
    result = truediv_backward(b, a, c)

    def fn(
        prev_grad: npt.NDArray[np.float32],
    ) -> tuple[
        Union[npt.NDArray[np.float32], Floatable],
        Union[npt.NDArray[np.float32], Floatable],
    ]:
        ret1, ret2 = result(prev_grad)
        # ret1 = cast(Union[npt.NDArray[np.float32], Floatable], ret1)
        ret2 = cast(Union[npt.NDArray[np.float32], Floatable], ret2)
        return ret2, ret1

    return fn
