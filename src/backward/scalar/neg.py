from src._types import GradFnScalar, Floatable


def neg_backward() -> GradFnScalar:

    def fn() -> tuple[Floatable, None]:
        return -1, None

    return fn
