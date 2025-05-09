from typing import Any, Callable, Union, cast, overload

import numpy as np
import numpy.typing as npt

from src.types import ArGradType, ArrayValueType, Floatable, GradFnArray, NumericDtypes

from .unbroadcast import unbroadcast


@overload
def pow_backward(
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
def pow_backward(
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
def pow_backward(
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


def pow_backward(
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

    mask_val = (a != 0) | np.zeros_like(result, dtype=np.bool_)
    mask_power = b == 1

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_a_temp = b * result / a, np.where(mask_power, 1.0, 0.0)  # type: ignore[operator]
        grad_a_temp = cast(npt.NDArray[Any], grad_a_temp)
        grad_a_raw = np.where(
            mask_val, grad_a_temp, np.where(mask_power, 1.0, 0.0)  # type: ignore[operator]
        ).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_b_raw = np.where(mask_val, result * np.log(a), 0.0).astype(np.float32)

    def fn(
        prev_grad: ArGradType,
    ) -> tuple[
        Union[ArGradType, Floatable],
        Union[ArGradType, Floatable],
    ]:
        grad_a = grad_a_raw * prev_grad
        grad_a = cast(ArGradType, grad_a)
        grad_a = unbroadcast(grad_a, a_shape)

        grad_b = grad_b_raw * prev_grad
        grad_b = cast(ArGradType, grad_b)
        grad_b = unbroadcast(grad_b, b_shape)

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
