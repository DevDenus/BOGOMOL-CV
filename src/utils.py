import numpy as np

# TODO: Apply effective matrix multiplication
def bin_matmul(mat_l : np.array, mat_r : np.array) -> np.array:
    """
    Implementation of binary matrix multiplication, using
    XOR and AND as summation and multiplication
    mat_l : np.array - left matrix
    mat_r : np.array - right matrix
    """
    assert mat_l.shape[1] == mat_r.shape[0], "Matrix dimensions doesn't match!"
    if not mat_l.dtype == mat_r.dtype == np.dtype('bool_'):
        mat_l, mat_r = np.array(mat_l, dtype=np.bool_), np.array(mat_r, dtype=np.bool_)
    result = np.array(
        [
            [
                np.count_nonzero(np.logical_and(mat_l[i, :], mat_r[:, j])) % 2 for j in range(mat_r.shape[1])
            ] for i in range(mat_l.shape[0])
        ],
        dtype=np.bool_
    )
    return result
