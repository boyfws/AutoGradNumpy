import numpy as np
from typing import Callable


def power_backward(
        val: float,
        power: float,
        calculated: float
) -> Callable[
    [],
    tuple[float, float]
]:
    if val == 0:
        val_grad = 0.0
    else:
        val_grad = power * calculated / val

    power_grad = calculated * np.log(val)

    def fn():
        return val_grad, power_grad

    return fn
