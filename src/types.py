from __future__ import annotations

from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Callable, Sequence, SupportsIndex, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.dtypes.Base import BaseArray


NotImplementedType = NotImplementedType

# -------------------- Types for Array --------------------
ArGradType = npt.NDArray[np.float32]

NpIndicesTypes = Union[
    slice,
    None,
    bool,
    SupportsIndex,
    npt.NDArray[np.integer[Any]],
    npt.NDArray[np.bool_],
    Sequence[bool],
    Sequence[int],
    Sequence[slice],
]


NumericDtypes = Union[
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
]

GradFnArray = Union[
    Callable[[ArGradType], tuple[ArGradType, ArGradType]],
    Callable[[ArGradType], tuple[ArGradType, None]],
]

ArrayValueType = npt.NDArray[Union[np.float16, np.float32, np.float64]]

# -------------------- Common types --------------------

Floatable = Union[int, float, NumericDtypes]

BaseOperationsType = Union[Floatable, npt.NDArray[NumericDtypes], "BaseArray"]
