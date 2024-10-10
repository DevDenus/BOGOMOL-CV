import numpy as np

def bin_vec2int(vector : np.array) -> int:
    result = int("".join(list(map(str, map(int, vector)))), 2)
    return result

def bin_mat_mul(weights : np.array, data : np.array) -> np.array:
    result = data @ weights.T
    
    return result
