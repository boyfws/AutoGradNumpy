from typing import Callable

from src.types import ArGradType


def neg_backward() -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        None,
    ],
]:

    def fn(prev_grad: ArGradType) -> tuple[ArGradType, None]:
        return -prev_grad, None

    return fn
