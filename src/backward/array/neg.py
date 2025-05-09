from src.types import ArGradType, GradFnArray


def neg_backward() -> GradFnArray:

    def fn(prev_grad: ArGradType) -> tuple[ArGradType, None]:
        return -prev_grad.copy(), None

    return fn
