from typing import Callable


def sub_backward(
        a: float,
        b: float,
        result: float
) -> tuple[
    Callable[[], float],
    Callable[[], float]
]:
    def fn1():
        return 1

    def fn2():
        return -1

    return fn1, fn2