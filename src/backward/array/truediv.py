from collections.abc import Callable
from typing import Union, cast, overload

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, GradFnArray, NumericDtypes

from .unbroadcast import unbroadcast


@overload
def truediv_backward(
    a: Union[ArrayValueType, npt.NDArray[NumericDtypes]],
    b: Union[ArrayValueType, npt.NDArray[NumericDtypes]],
    result: npt.NDArray[np.float_],
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        ArGradType,
    ],
]: ...


@overload
def truediv_backward(
    a: ArrayValueType,
    b: Floatable,
    result: npt.NDArray[np.float_],
) -> Callable[
    [ArGradType],
    tuple[
        ArGradType,
        Floatable,
    ],
]: ...


@overload
def truediv_backward(
    a: Floatable,
    b: ArrayValueType,
    result: npt.NDArray[np.float_],
) -> Callable[
    [ArGradType],
    tuple[
        Floatable,
        ArGradType,
    ],
]: ...


def truediv_backward(
    a: Union[npt.NDArray[NumericDtypes], Floatable],
    b: Union[npt.NDArray[NumericDtypes], Floatable],
    result: npt.NDArray[np.float_],
) -> Union[
    GradFnArray,
    Callable[
        [ArGradType],
        tuple[
            Floatable,
            ArGradType,
        ],
    ],
]:

    a_array_flag = isinstance(a, np.ndarray)
    b_array_flag = isinstance(b, np.ndarray)

    a_shape = a.shape if a_array_flag else None
    b_shape = b.shape if b_array_flag else None

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[
        Union[ArGradType, Floatable],
        Union[ArGradType, Floatable],
    ]:
        grad_a_raw = (prev_grad / b).astype(np.float32)  # type: ignore[operator]
        grad_a = unbroadcast(grad_a_raw, a_shape)

        grad_b_raw = (-prev_grad * result / b).astype(np.float32)
        grad_b = unbroadcast(grad_b_raw, b_shape)

        return grad_a, grad_b

    if a_array_flag and b_array_flag:
        return cast(
            Callable[
                [ArGradType],
                tuple[
                    ArGradType,
                    ArGradType,
                ],
            ],
            fn,
        )

    elif a_array_flag:
        return cast(
            Callable[
                [ArGradType],
                tuple[
                    ArGradType,
                    Floatable,
                ],
            ],
            fn,
        )

    elif b_array_flag:
        return cast(
            Callable[
                [ArGradType],
                tuple[
                    Floatable,
                    ArGradType,
                ],
            ],
            fn,
        )

    else:
        raise ValueError("Wrong input")
