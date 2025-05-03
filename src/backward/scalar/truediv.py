from typing import Callable


def truediv_backward(
        num: float,
        den: float,
        result: float
) -> tuple[
    Callable[[], float],
    Callable[[], float]
]:
    def num_fn():
        return 1 / den

    def den_fn():
        return -result / den

    return num_fn, den_fn


