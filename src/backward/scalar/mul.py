from typing import Callable


def mul_backward(
        a: float,
        b: float,
        result: float
) -> tuple[
    Callable[[], float],
    Callable[[], float]
]:

    def fn_a():
        return b

    def fn_b():
        return a

    return fn_a, fn_b