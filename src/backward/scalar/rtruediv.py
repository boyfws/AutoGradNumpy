from typing import cast

from src.types import Floatable, GradFnScalar

from .truediv import truediv_backward


def rtruediv_backward(a: Floatable, b: Floatable, c: Floatable) -> GradFnScalar:
    result = truediv_backward(b, a, c)

    def fn() -> tuple[
        Floatable,
        Floatable,
    ]:
        ret1, ret2 = result()
        ret1 = cast(Floatable, ret1)
        ret2 = cast(Floatable, ret2)
        return ret2, ret1

    return fn
