import numpy as np
from src._types import Floatable, GradFnScalar


def power_backward(
        val: Floatable,
        power: Floatable,
        calculated: Floatable
) -> GradFnScalar:
    if val == 0:
        val_grad = 0.0
    else:
        val_grad = power * calculated / val  # type: ignore[operator]

    power_grad = calculated * np.log(val)

    def fn() -> tuple[Floatable, Floatable]:
        return val_grad, power_grad

    return fn
