def sub_backward(a, b, result):
    def fn1():
        return 1

    def fn2():
        return -1

    return fn1, fn2