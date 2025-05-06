from typing import (
    Union,
    Callable,
    TypeAlias,
    Any
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
                Union[Floatable, np.ndarray[Any, np.float32], None],
                Union[Floatable, np.ndarray[Any, np.float32], None],
            ]
        ]

GradFnArray = Callable[
    [np.ndarray[Any, np.ndarray[Any, np.float32]]],
    tuple[
        Union[Floatable, np.ndarray[Any, np.float32], None],
        Union[Floatable, np.ndarray[Any, np.float32], None],
    ]
]

NotImplementedType: TypeAlias = type(NotImplemented)