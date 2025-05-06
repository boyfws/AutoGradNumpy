from src._types import GradFnScalar, Floatable


def mul_backward(
        a: Floatable,
        b: Floatable,
        result: Floatable
) -> GradFnScalar:

    def fn() -> tuple[
        Floatable,
        Floatable
    ]:
        return b, a

    return fn
