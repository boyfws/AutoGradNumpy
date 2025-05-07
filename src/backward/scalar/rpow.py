from typing import cast

from src.types import Floatable, GradFnScalar

from .power import power_backward


def rpow_backward(a: Floatable, b: Floatable, c: Floatable) -> GradFnScalar:
    result = power_backward(b, a, c)

    def fn() -> tuple[
        Floatable,
        Floatable,
    ]:
        ret1, ret2 = result()
        ret1 = cast(Floatable, ret1)
        ret2 = cast(Floatable, ret2)
        return ret2, ret1

    return fn
