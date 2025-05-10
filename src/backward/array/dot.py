from typing import Callable
from src.types import ArrayValueType, NumericDtypes, ArGradType
import numpy as np
import numpy.typing as npt


def dot_backward(
        a: ArrayValueType,
        b: npt.NDArray[NumericDtypes],
        c: npt.NDArray[np.float_]
) -> Callable[[ArGradType], tuple[ArGradType, ArGradType]]:

    def fn(prev_grad: ArGradType) -> tuple[ArGradType, ArGradType]:
        return prev_grad.dot(b.T).astype(np.float32), a.T.dot(prev_grad).astype(np.float32)

    return fn