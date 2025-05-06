from typing import (
    Union,
    Callable,
    TypeAlias
)
import numpy as np

Floatable = Union[
    int,
    float,
    np.generic
]

GradFnScalar = Callable[
            [],
            tuple[
                Union[Floatable, np.ndarray, None],
                Union[Floatable, np.ndarray, None],
            ]
        ]

NotImplementedType: TypeAlias = type(NotImplemented)