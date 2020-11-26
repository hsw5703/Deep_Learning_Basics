# 신경망학습: 미니 배치(mini-batch)
import os
import sys
import numpy as np
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from common import cross_entropy_error

except ImportError:
    print('Library Module Can Not Found')

# test1
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
print(train_x.shape)  # 60000 x 784
print(train_t.shape)  # 60000 x 10

train_size = len(train_x)  # 60000
batch_size = 10

batch_mask = np.random.choice(train_size, batch_size)  # train_size(60000개) 중에서 batch_size(10개)를 임의로 뽑는다.
print(batch_mask)

train_x_batch = train_x[batch_mask]  # batch_mask를 행의 index로 하여 데이터를 선별한다.
train_t_batch = train_t[batch_mask]
print(train_x_batch.shape)  # 10 X 784
print(train_t_batch.shape)  # 10 x 10

# test2
# 만약에 batch_size가 3인 경우
t = np.array([
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
])

y = np.array([
    [0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.],
    [0.1, 0.05, 0., 0.4, 0.02, 0.03, 0.1, 0.3, 0., 0.],
    [0., 0.92, 0.02, 0., 0.02, 0.03, 0.01, 0., 0., 0.]
])

print(cross_entropy_error(y, t))
# cross_entropy_error() 함수에서 오차 e를 구할 때 batch_size로 나눠주는 이유는 np.sum()을 axis에 따라
# 나눠서 진행하지 않기 때문이고 이는 전체 y에 대한 오차를 구하고 싶기 때문이다.
# 각각의 y에 대한 오차를 구한다면 np.sum(axis=1)을 해주고 batch_size로 나눠주지 않으면 된다.
# 이렇게 batch로 나눠주는 이유는 훈련 데이터의 일부만 무작위로 골라 훈련시켜 전체의 근사치로 이용하기 위해서다.
# 데이터 전체를 사용해 손실함수 e를 구하는 건 현실적이지 않다. 시간이 오래 걸림.
