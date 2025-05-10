from typing import Any


class EmptyCallable:
    """
    The class of singleton to which grad_fn will be replaced in
    the components of the computational graph at the backpropagation pass through it,
    we can consider this class as a marker that the backpropagation pass has been done
    """

    def __call__(self, prev_grad: Any) -> tuple[None, None]:
        return None, None
