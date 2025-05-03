import numpy as np


def power_backward(val, power, calculated):
    def val_fn():
        if val == 0:
            return 0

        return power * calculated / val

    def power_fn():
        return calculated * np.log(val)

    return val_fn, power_fn