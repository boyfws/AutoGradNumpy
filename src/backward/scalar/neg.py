from typing import Callable


def neg_backward() -> Callable[
    [],
    tuple[float, None]
]:

    def fn():
        return -1, None

    return fn