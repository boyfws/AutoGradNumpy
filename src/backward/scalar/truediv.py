from src._types import Floatable, GradFnScalar


def truediv_backward(
        num: Floatable,
        den: Floatable,
        result: Floatable
) -> GradFnScalar:

    def fn() -> tuple[Floatable, Floatable]:
        return 1 / den, -result / den  # type: ignore[operator]

    return fn


