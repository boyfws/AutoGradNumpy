import numpy as np

from .Scalar import Scalar


class Float16(Scalar):
    _dtype = np.float16
