# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification): 배치처리
# batch 처리는 자료를 모아 두었다가 일괄해서 처리하는 자료처리의 형태를 뜻한다.
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
    from common import sigmoid, softmax
except ImportError:
    print('Library Module Can Not Found')

# 1. 매개변수(w, b) 데이터 셋 가져오기
network = init_network()

w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

# 2. 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

# 3. 정확도 산출
hit = 0
xlen = len(test_x) # 10000
batch_size = 100

for idx, batch_sidx in enumerate(range(0, xlen, batch_size)): # 1부터 10000까지의 데이터를 100개 단위로 나누어 연산.
    batch_x = test_x[batch_sidx:batch_sidx+batch_size] # 0~99 / 100~199 등등 # (100, 784) tensor 2

    a1 = np.dot(batch_x, w1) + b1 # (100, 784) * (784 * 50) = (100 * 50)
    z1 = sigmoid(a1) # 은닉층 활성 함수

    a2 = np.dot(z1, w2) + b2 # (100 * 50) * (50 * 100) = (100 * 100)
                             # b2는 100 vector이지만 Broadcasting되어 100행으로 늘어나 더해진다.
    z2 = sigmoid(a2) # 은닉층 활성 함수
    a3 = np.dot(z2, w3) + b3 # (100 * 100) * (100 * 10) = (100 * 10)
    batch_y = softmax(a3) # 출력층 활성 함수
    batch_predict = np.argmax(batch_y, axis=1)
    # batch_y (100 * 10) matrix는 100개의 데이터가 10개의 열을 가지는데 10개의 열의 합은 1이다.
    # 또 각각의 index는 이미지가 나타내는 숫자를 의미하고 데이터 값은 그 숫자일 확률을 의미한다.
    # 고로 각 행의 최댓값의 index를 반환하는 argmax()를 이용하면 예측한 값을 알 수 있다.

    batch_t = test_t[batch_sidx:batch_sidx+batch_size]

    batch_hit = np.sum(batch_predict == batch_t)
    # 'batch_predict == batch_t'를 출력하면 True / False 값이 반환되고
    # 이를 np.sum()해주면 batch 안의 데이터 중 올바르게 예측한 숫자를 알 수 있다.
    hit += batch_hit

    print(f'batch #{idx+1}, batch hit: {batch_hit}, total hit:{hit}')


# 정확도(Accuracy)
print(f'Accuracy: {hit/xlen}')
