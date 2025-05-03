def truediv_backward(
        num, den, result
):
    def num_fn():
        return 1 / den

    def den_fn():
        return -result / den

    return num_fn, den_fn


