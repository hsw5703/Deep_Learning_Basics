import numpy as np

# sigmoid function

def sigmoid(x) :
    return 1 / (1 + np.e**(-x))

def relu(x) :
    return np.maximum(0, x)
