from src.types import Floatable, GradFnScalar


def mul_backward(a: Floatable, b: Floatable, result: Floatable) -> GradFnScalar:

    def fn() -> tuple[Floatable, Floatable]:
        return b, a

    return fn
