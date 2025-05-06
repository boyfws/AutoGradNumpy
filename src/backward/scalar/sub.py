from src._types import Floatable, GradFnScalar


def sub_backward(
        a: Floatable,
        b: Floatable,
        result: Floatable
) -> GradFnScalar:

    def fn() -> tuple[Floatable, Floatable]:
        return 1.0, -1.0

    return fn
