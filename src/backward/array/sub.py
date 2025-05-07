from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt

from src.types import ArrayValueType, Floatable, GradFnArray

from .add import add_backward


def sub_backward(
    a: ArrayValueType, b: Union[npt.NDArray[Any], Floatable], result: npt.NDArray[Any]
) -> GradFnArray:

    target_fn = add_backward(a, b, result)

    def fn(
        prev_grad: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], Union[npt.NDArray[np.float32], Floatable]]:
        res = target_fn(prev_grad)
        if res[1] is not None:
            ret_1 = -res[1]  # type: ignore[operator]
            ret_1 = cast(Union[npt.NDArray[np.float32], Floatable], ret_1)
        else:
            ret_1 = None

        return res[0], ret_1

    return fn
