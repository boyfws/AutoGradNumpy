from typing import Optional, Union

from src.dtypes import Float16, Float32, Float64
from src.dtypes.Base import BaseScalar, BaseArray
import numpy as np


class Array(BaseArray):
    _ret_scalar_dtype: BaseScalar
    _dtype: Union[np.float16, np.float32, np.float64]

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

        self._array = array.copy().astype(self._dtype)

        self.requires_grad = requires_grad