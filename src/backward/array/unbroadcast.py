from typing import Literal, Optional, Union, overload

from src.types import ArGradType, Floatable


@overload
def unbroadcast(grad: ArGradType, shape: Literal[None]) -> Floatable: ...


@overload
def unbroadcast(grad: ArGradType, shape: tuple[int, ...]) -> ArGradType: ...


def unbroadcast(
    grad: ArGradType, shape: Optional[tuple[int, ...]] = None
) -> Union[ArGradType, Floatable]:
    """
    Reduce `grad` back to `shape` by summing over broadcasted dimensions.
    """
    if shape is None:
        return grad.sum().reshape(())

    ndim_diff = grad.ndim - len(shape)
    shape_padded = (1,) * ndim_diff + shape  # e.g. (1,1,3,4)

    axes = tuple(i for i, dim in enumerate(shape_padded) if dim == 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)
