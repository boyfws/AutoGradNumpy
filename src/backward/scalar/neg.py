from src.types import Floatable, GradFnScalar


def neg_backward() -> GradFnScalar:

    def fn() -> tuple[Floatable, None]:
        return -1, None

    return fn
