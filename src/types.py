from types import NotImplementedType
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

Floatable = Union[int, float, np.generic]

GradEl = Union[Floatable, npt.NDArray[np.float32]]

GradFnScalar = Callable[
    [],
    tuple[
        GradEl,
        Optional[GradEl],
    ],
]

GradFnArray = Callable[
    [npt.NDArray[np.float32]],
    tuple[
        GradEl,
        Optional[GradEl],
    ],
]

ArrayValueType = npt.NDArray[Union[np.float16, np.float32, np.float64]]

NotImplementedType = NotImplementedType
