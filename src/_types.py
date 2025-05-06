from typing import (
    SupportsFloat,
    Union,
    Callable
)
import numpy as np

Floatable = Union[SupportsFloat, np.generic]

GradFnScalar = Callable[
            [],
            tuple[
                Union[float, np.ndarray, None],
                Union[float, np.ndarray, None],
            ]
        ]

NotImplementedType = type(NotImplemented)