def mul_backward(a, b, result):

    def fn_a():
        return b

    def fn_b():
        return a

    return fn_a, fn_b