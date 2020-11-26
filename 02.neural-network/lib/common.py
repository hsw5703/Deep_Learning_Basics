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


# softmax activation function: 오버플로우 대책 & 배치처리지원 수정
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    # x = x - np.max(x, axis=1)
    # y = np.exp(x) / np.sum(np.exp(x), axis=1)
    # return y
    # 전치를 하지 않았을 경우에는 Error가 발생한다. 이는 matrix와 vector의 합의 특징 때문인데,
    # vector를 matrix에 더해줄 때에는 matrix의 열의 수와 vector의 크기가 같아야 한다.
    # 만약 (3 * 5) matrix와 3 vector를 더한다면 열의 수와 맞지 않아 Error가 발생한다.
    # 하지만 5 vector를 더하면 열의 수와 맞아 5 vector가 늘어나 (3 * 5) matrix 형태로 행렬의 합을 실시한다.
    # 위의 경우도 마찬가지로, 전치를 하지 않을 경우 x는 (100 * 10), np.max(x, axis=0)은 100 vector의 형태로 나온다.
    # 이 때 열은 10 vector는 100으로 맞지 않아 Error가 발생하므로 전치를 해줘야 한다.

    x = x - np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    return y


# Sum of Squares Error(SSE)
def sum_squares_error(y, t):
    e = 0.5 * np.sum((y-t)**2)
    return e


# batch 없을 때
# cross entropy error
# t = one hot
def cross_entropy_error_non_batch(y, t): # delta에 작은 값을 넣어준 이유는 log에 0이 들어갔을 경우 무한대로 오차가 커지는 걸 방지하기 위해서이다.
    delta = 1.e-7
    e = -np.sum(t * np.log(y+delta))
    return e


# cross entropy error
# t = one hot
# for batch
def cross_entropy_error(y, t): # y.ndim == 2인 경우는 ex03에서 확인할 수 있다.
    if y.ndim == 1 :
        y = y.reshape(1, y.size) # (1 * y.size) matrix
        t = t.reshape(1, t.size) # (1 * t.size) matrix

    batch_size = y.shape[0]

    delta = 1.e-7
    e = -np.sum(t * np.log(y+delta)) / batch_size

    return e

def numerical_diff1(f, w, x, t):
    h = 1e-4
    gradient = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    # np.nditer는 배열을 for문을 사용하지 않고 순서대로 index를 출력한다.
    # flags = ['multi_index'] 는 2차원 또는 3차원의 형태로 위치를 튜플로 받겠다는 뜻.
    # op_flags=['readwrite'] 는 읽고 쓰기를 같이 하겠다는 뜻.
    while not it.finished: # 경사하강법
        idx = it.multi_index
        temp = w[idx] # op_flags=['readwrite']의 읽기 부분.

        w[idx] = temp + h # op_flags=['readwrite']의 쓰기 부분.
        h1 = f(w, x, t)

        w[idx] = temp - h
        h2 = f(w, x, t)

        gradient[idx] = (h1 - h2) / (2 * h)

        w[idx] = temp
        it.iternext() # 다음 순서로 넘어가게 한다.

    return gradient


numerical_gradient1 = numerical_diff1


def numerical_diff2(f, w):
    h = 1e-4
    gradient = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        temp = w[idx]

        w[idx] = temp + h
        h1 = f(w)

        w[idx] = temp - h
        h2 = f(w)

        gradient[idx] = (h1 - h2) / (2 * h)

        w[idx] = temp   # 값복원
        it.iternext()

    return gradient


numerical_gradient2 = numerical_diff2