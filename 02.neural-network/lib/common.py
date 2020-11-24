import numpy as np

# sigmoid function

def sigmoid(x) :
    return 1 / (1 + np.e**(-x))

def relu(x) :
    return np.maximum(0, x)

def identity(x):
    return x

# softmax activation function: 큰값에서 NAN 반환하는 불안한 함수
def softmax_overflow(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# softmax activation function: 오버플로우 대책으로 수정함수
def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

