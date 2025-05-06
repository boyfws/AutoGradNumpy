class _EmptyCallable:
    """
    The class of singleton to which grad_fn will be replaced in
    the components of the computational graph at the backpropagation pass through it,
    we can consider this class as a marker that the backpropagation pass has been done
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True

    def __call__(self) -> tuple[None, None]:
        return None, None
