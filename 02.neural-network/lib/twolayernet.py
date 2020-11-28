# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid, softmax, cross_entropy_error, numerical_gradient2
except ImportError:
    print('Library Module Can Not Found')

params = dict()

# 앞선 '05.mnist_~'에서는 학습된 가중치와 편향을 from mnist import init_network 에서 가져와 사용했지만
# 이번 예제에서는 모두 학습부터 시작한다.

def initialize(sz_input, sz_hidden, sz_output, w_init=0.01) : # 초기값 설정
    params['w1'] = w_init * np.random.randn(sz_input, sz_hidden)
    # np.random.randn(sz_input, sz_hidden) = 평균이 0이고 표준편차가 1인 (784, 50)의 martrix를 생성한다.
    # 이 때 w_init을 곱해주어 적당히 작은 값으로 설정하는데 이 값은 학습을 하기 위해 설정하는 초기값일 뿐 어떤 의미도 없다.
    params['b1'] = np.zeros(sz_hidden) # 50 vector 생성
    params['w2'] = w_init * np.random.randn(sz_hidden, sz_output) # (50, 10) matrix
    params['b2'] = np.zeros(sz_output) # 10 vector 생성
    # zeros() 대신 zeros_like()를 쓰면 안되는 이유는 ex05에서 연산할 때 float과 int가 같이 연산되어 충돌이 일어나기 때문이다.
    # zeros()는 float형이지만 zeros_like()는 int형임.

def forward_progation(x):
    w1 = params['w1']
    b1 = params['b1']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    w2 = params['w2']
    b2 = params['b2']
    a2 = np.dot(z1, w2) + b2

    y = softmax(a2)
    return y


def loss(x, t):
    y = forward_progation(x)
    e = cross_entropy_error(y, t)
    return e


def accuracy(x, t) :
    y = forward_progation(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc

def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            temp = param[idx]

            param[idx] = temp + h
            h1 = loss(x, t)

            param[idx] = temp - h
            h2 = loss(x, t)

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = temp   # 값복원
            it.iternext()

        gradient[key] = param_gradient

    return gradient
