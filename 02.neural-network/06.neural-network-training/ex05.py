import os
import pickle
import sys
import time

import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1.load training/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2.hyperparameters
numiters = 1  # 10000
szbatch = 100
sztrain = train_x.shape[0] # 60000
ratelearning = 0.1

# train_x = (60000 * 784)
# train_t = (60000 * 10)

# 3.initialize network
network.initialize(sz_input=train_x.shape[1], sz_hidden=50, sz_output=train_t.shape[1])
# sz_input은 입력층의 열의 수를 의미한다.
# sz_hidden은 은닉층의 데이터 수를 의미한다.


# 4.training
train_losses = []

for idx in range(numiters):

    # 4-1. fetch mini-batch
    batch_mask = np.random.choice(sztrain, szbatch) # sztrain(60000) 중 szbatch(100)의 index를 random 추출.
    # np.random.choice(['b', 'c'], 10, p=[0.9, 0.1]) 'b'와 'c'를 10개를 랜덤 추출하는데 이때 'b'는 90% 확률로, 'c'는 10% 확률로 추출한다.
    train_x_batch = train_x[batch_mask] # (100, 784)
    train_t_batch = train_t[batch_mask] # (100, 10)

    # 4-2. gradient
    stime = time.time()             # stopwatch: start
    gradient = network.numerical_gradient_net(train_x_batch, train_t_batch)
    # training data 6만 개 중 100개를 임의로 뽑아 경사하강법으로 학습시킨다.
    elapsed = time.time() - stime   # stopwatch: end

    # 4-3. update parameter
    for key in network.params:
        network.params[key] -= ratelearning * gradient[key]

    # 4-4. train loss
    loss = network.loss(train_x_batch, train_t_batch)
    train_losses.append(loss)

    print(f'#{idx+1}: loss:{loss}, elapsed time: {elapsed}s')

# serialize train loss : 학습 결과를 pickle 형태로 남겨두는 것.
train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer-train-loss.pkl')
# C:\Deep-Learning\Deep_Learning_Basics\02.neural-network\06.neural-network-training\dataset\twolayer-train-loss.pkl
# 위와 같이 주소를 이어붙여 저장한다.
print(f'Save Pickle({train_loss_file}) file....')
with open(train_loss_file, 'wb') as f:
    pickle.dump(train_losses, f, -1) # 손실을 저장한 train_losses 변수를 f 파일에 저장한다.
print('Done!')