from .Scalar import Scalar
import numpy as np


class Float16(Scalar):
    _dtype = np.float16