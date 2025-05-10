from typing import Any, Type, Optional


class EmptyCallable:
    """
    The class of singleton to which grad_fn will be replaced in
    the components of the computational graph at the backpropagation pass through it,
    we can consider this class as a marker that the backpropagation pass has been done
    """

    _instance: Optional["EmptyCallable"] = None

    def __new__(
        cls: Type["EmptyCallable"],
        *args: Any,
        **kwargs: Any
    ) -> "EmptyCallable":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True
