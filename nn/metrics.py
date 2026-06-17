import numpy as np

def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)
