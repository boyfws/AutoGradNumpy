from typing import Callable


def truediv_backward(
        num: float,
        den: float,
        result: float
) -> Callable[
    [],
    tuple[
        float,
        float
    ]
]:

    def fn():
        return 1 / den, -result / den

    return fn


