from typing import Callable


def add_backward(
        a: float,
        b: float,
        result: float
) -> tuple[
    Callable[[], float], Callable[[], float]
]:
    def fn():
        return 1
    return fn, fn
