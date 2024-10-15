import numpy as np

# TODO: Apply effective matrix multiplication using
# XOR and AND as summation and multiplication
def bin_matmul(mat_l : np.array, mat_r : np.array) -> np.array:
    """
    Implementation of binary matrix multiplication, using
    XOR and AND as summation and multiplication
    mat_l : np.array - left matrix
    mat_r : np.array - right matrix
    """
    assert mat_l.shape[1] == mat_r.shape[0], "Matrix dimensions doesn't match!"
    if not mat_l.dtype == mat_r.dtype == np.dtype('int_'):
        mat_l, mat_r = np.array(mat_l, dtype=np.int_), np.array(mat_r, dtype=np.int_)
    result = mat_l @ mat_r % 2
    return np.array(result, dtype=np.bool_)
