from src._types import GradFnScalar, Floatable


def add_backward(
        a: Floatable,
        b: Floatable,
        result: Floatable
) -> GradFnScalar:

    def fn() -> tuple[
        Floatable,
        Floatable,
    ]:
        return 1.0, 1.0

    return fn
