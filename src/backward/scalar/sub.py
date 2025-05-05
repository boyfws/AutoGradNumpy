from typing import Callable


def sub_backward(
        a: float,
        b: float,
        result: float
) -> Callable[
    [],
    tuple[float, float]
]:
    def fn():
        return 1, -1

    return fn