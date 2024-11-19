import numpy as np

def bin_threshold_function(x : np.array, threshold : int = 0):
    """
    Returns 1 if x > threshold, returns 0 elsewhere
    """
    summed_value = np.sum(x, axis=1)
    return summed_value > threshold
