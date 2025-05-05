from typing import Callable


def mul_backward(
        a: float,
        b: float,
        result: float
) -> Callable[
    [],
    tuple[float, float]
]:

    def fn():
        return b, a

    return fn