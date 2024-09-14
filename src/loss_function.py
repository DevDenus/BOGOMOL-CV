import numpy as np

def categorical_crossentropy(y_pred : np.array, y_true : np.array) -> np.array:
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return np.sum(y_true * -np.log(y_pred), axis=-1, keepdims=False)
