from types import NotImplementedType
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

Floatable = Union[int, float, np.generic]

GradFnScalar = Callable[
    [],
    tuple[
        Union[Floatable, npt.NDArray[np.float32], None],
        Union[Floatable, npt.NDArray[np.float32], None],
    ],
]

GradFnArray = Callable[
    [npt.NDArray[np.float32]],
    tuple[
        Union[Floatable, npt.NDArray[np.float32], None],
        Union[Floatable, npt.NDArray[np.float32], None],
    ],
]

NotImplementedType = NotImplementedType
