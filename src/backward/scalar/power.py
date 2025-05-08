import numpy as np

from src.types import Floatable, GradFnScalar


def power_backward(
    val: Floatable, power: Floatable, calculated: Floatable
) -> GradFnScalar:
    if val != 0:
        val_grad = power * calculated / val  # type: ignore[operator]
        power_grad = calculated * np.log(val)
    else:
        if power == 1:
            val_grad = 1.0
        else:
            val_grad = 0.0
        power_grad = 0.0

    def fn() -> tuple[Floatable, Floatable]:
        return val_grad, power_grad

    return fn
