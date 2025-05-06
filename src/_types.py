from typing import (
    Union,
    Callable,
    TypeAlias,
    Any
)
import numpy as np
import numpy.typing as npt

Floatable = Union[
    int,
    float,
    np.generic
]

GradFnScalar = Callable[
            [],
            tuple[
                Union[Floatable, npt.NDArray[np.float32], None],
                Union[Floatable, npt.NDArray[np.float32], None],
            ]
        ]

GradFnArray = Callable[
    [npt.NDArray[np.float32]],
    tuple[
        Union[Floatable, npt.NDArray[np.float32], None],
        Union[Floatable, npt.NDArray[np.float32], None],
    ]
]

NotImplementedType: TypeAlias = type(NotImplemented)