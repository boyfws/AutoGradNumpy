import torch
import numpy as np


def compare_tensor_ops(fn_arr, fn_torch, name=""):
    """
    Helper: compare outputs and gradients for array_test vs torch.Tensor ops.
    """
    # array_test
    arr_res, arr_inputs = fn_arr()
    # if output is scalar, use scalar grad, else ones_like
    arr_res.sum().backward()
    assert arr_res._grad is None

    assert not arr_res.is_leaf

    arr_out = arr_res.data

    # torch
    t_res, t_inputs = fn_torch()
    t_res.sum().backward()


    assert t_res.requires_grad
    assert not t_res.is_leaf

    t_out = t_res.detach().cpu().numpy()

    arr_grads = []
    t_grads = []
    for i in range(len(t_inputs)):
        assert t_inputs[i].is_leaf
        assert arr_inputs[i].is_leaf
        if arr_inputs[i].grad is not None and t_inputs[i].grad is not None:
            arr_grads.append(arr_inputs[i].grad)
            t_grads.append(t_inputs[i].grad.cpu().numpy())

    assert len(arr_grads) > 0
    # compare outputs
    assert np.allclose(arr_out, t_out), f"Output mismatch in {name}: {arr_out} vs {t_out}"
    # compare grads
    for i, (ag, tg) in enumerate(zip(arr_grads, t_grads)):
        assert np.allclose(ag, tg), f"Grad #{i} mismatch in {name}: {ag} vs {tg}"