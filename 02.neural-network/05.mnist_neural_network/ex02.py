# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification): 신호전달 I
# 임의의 한 데이터만 뽑아 예측 값과 실제 값을 출력
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

# 2. 학습/시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

xlen = len(test_x) # 10000개의 test 데이터가 있으므로 xlen = 10000
randidx = np.random.randint(0, xlen, 1).reshape(()) # reshape(())은 vector 형식을 scalar 형식으로 바꿔준다.
                                                    # 0부터 xlen 미만의 숫자 중 하나를 무작위로 뽑는다.
# 3. 신호전달
print('\n== 신호전달 구현1: 은닉 1층 전달 ============================')

x = test_x[randidx]
print(f'x dimension: {x.shape}')        # 784 vector # reshape(())를 해주지 않으면 1 * 784 matrix

w1 = network['W1']
print(f'w1 dimension: {w1.shape}')      # 784 x 50 matrix
b1 = network['b1']
print(f'b1 dimension: {b1.shape}')      # 50 vector
a1 = np.dot(x, w1) + b1 # vector와 matrix를 곱해주면(dot) vector가 나온다.
print(f'a1 = {a1}') # 50 vector

print('\n== 신호전달 구현2: 은닉 1층 활성함수 h() 적용 ============================')

print(f'a1 dimension: {a1.shape}')      # 50 vector
z1 = sigmoid(a1)
print(f'z1 = {z1}')

print('\n== 신호전달 구현3: 은닉 2층 전달 ============================')

print(f'z1 dimension: {z1.shape}')      # 50 vector
w2 = network['W2']
print(f'w2 dimension: {w2.shape}')      # 50 x 100 matrix
b2 = network['b2']
print(f'b2 dimension: {b2.shape}')      # 100 vector
a2 = np.dot(z1, w2) + b2
print(f'a2 = {a2}') # 100 vector

print('\n== 신호전달 구현4: 은닉 2층 활성함수 h() 적용 ============================')

print(f'a2 dimension: {a2.shape}')      # 100 vector
z2 = sigmoid(a2)
print(f'z2 = {z2}')

print('\n== 신호전달 구현5: 출력층 전달 ============================')

print(f'z2 dimension: {z2.shape}')      # 100 vector
w3 = network['W3']
print(f'w3 dimension: {w3.shape}')      # 100 x 10 matrix
b3 = network['b3']
print(f'b3 dimension: {b3.shape}')      # 10 vector
a3 = np.dot(z2, w3) + b3
print(f'a3 = {a3}')

print('\n== 신호전달 구현6: 출력층 활성함수 σ() 적용 =======================')
print(f'a3 dimension: {a3.shape}')      # 10 vector
y = softmax(a3) # 10 vector, index = 0~9는 각각 이미지가 0일 확률, 1일 확률 등을 나타낸다.
print(f'y = {y}') # y의 10개 데이터의 합은 1이다.

print('\n== 예측 결과 ======================================')
predict = np.argmax(y) # 최댓값의 index를 반환한다. np.argmax(y, axis = 0)은 배열의 경우 0열끼리, 1열끼리 한줄씩 비교한다.
print(f'{randidx+1} 번째 이미지 예측: {predict}')

print('\n== 정답 ======================================')
t = test_t[randidx]
print(f'{randidx+1} 번째 이미지 레이블: {t}') # 실제 들어있는 데이터의 숫자