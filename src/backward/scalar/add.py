def add_backward(a, b, result):
    def fn():
        return 1
    return fn, fn
