import numpy as np

def bin_threshold_function(x : np.array, threshold : int = 0):
    """
    Returns 1 if x > threshold, returns 0 elsewhere
    """
    summed_value = np.array(list(map(int.bit_count, map(int, x))))
    return summed_value > threshold
