from typing import Optional, Union

from src.dtypes import Float16, Float32, Float64
from src.dtypes.Base import BaseScalar, BaseArray
import numpy as np


class Array(BaseArray):
    _ret_scalar_dtype: BaseScalar
    _dtype: Union[np.float16, np.float32, np.float64]
    prev_1: ...
    prev_2: ...
    grad: Optional[float]
    grad_fn: ...

    def __init__(self,
                 array,
                 dtype: Optional[
                     Union[np.float16, np.float32, np.float64]
                 ] = None,
                 requires_grad: bool = False,
                 ):
        if dtype is None:
            self._dtype = np.float32

        else:
            if dtype == np.float16:
                self._dtype = np.float16
                self._ret_scalar_dtype = Float16
            elif dtype == np.float32:
                self._dtype = np.float32
                self._ret_scalar_dtype = Float32
            elif dtype == np.float64:
                self._dtype = np.float64
                self._ret_scalar_dtype = Float64
            else:
                raise TypeError("dtype must be np.float16, np.float32, or np.float64")

        self.array = array.copy().astype(self._dtype)

        self.requires_grad = requires_grad
        self.prev_1 = None
        self.prev_2 = None
        self.grad = None
        self.grad_fn = None

    def __str__(self) -> str:
        return str(self.array)

    def item(self) -> np.ndarray:
        return self.array.copy()

    def __getitem__(self, *args, **kwargs) -> np.ndarray | float:
        return self.array.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs) -> None:
        self.requires_grad = False

        if self.grad is not None:
            self.grad = np.zeros_like(self.array, dtype=np.float32)

        self.array.__setitem__(*args, **kwargs)

    def _backward(self, prev_grad: float) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def detach(self) -> "BaseScalar":
        pass

    def __neg__(self) -> "BaseScalar":
        pass

    def __add__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __mul__(self,
                other
                ):
        pass

    def __rmul__(self, other):
        pass


    def __truediv__(self, other):
        pass

    def __rtruediv__(self):
        pass

    def __pow__(
            self,
            power
    ):
        pass

    def __rpow__(self, other):
        pass

    def __eq__(self, other):
        pass

    def sum(self):
        pass

    def abs(self):
        pass

