import numpy as np
from typing import Callable


def power_backward(
        val: float,
        power: float,
        calculated: float
) -> tuple[
    Callable[[], float],
    Callable[[], float]
]:
    def val_fn():
        if val == 0:
            return 0

        return power * calculated / val

    def power_fn():
        return calculated * np.log(val)

    return val_fn, power_fn
