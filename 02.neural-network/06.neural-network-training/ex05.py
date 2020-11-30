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
numiters = 10  # 10000
szbatch = 100
sztrain = train_x.shape[0] # 60000
szepoch = sztrain/szbatch   # 전체 학습 데이터로 학습을 끝마쳤을 때 -> 1epoch: 60,000 / 200 = 300
ratelearning = 0.1

# train_x = (60000 * 784)
# train_t = (60000 * 10)

# 3.initialize network
network.initialize(sz_input=train_x.shape[1], sz_hidden=50, sz_output=train_t.shape[1])
# sz_input은 입력층의 열의 수를 의미한다.
# sz_hidden은 은닉층의 데이터 수를 의미한다.


# 4.training
train_losses = []
train_accuracies = []
test_accuracies = []

for idx in range(numiters+1):

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

    # 4-5 accuracy per epoch
    if idx % szepoch == 0:
        train_accuracy = network.accuracy(train_x, train_t)
        train_accuracies.append(train_accuracy)

        test_accuracy = network.accuracy(test_x, test_t)
        test_accuracies.append(test_accuracy)

    print(f'#{idx}: loss:{loss}, elapsed time: {elapsed}s')

# 5. serialization
params_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_params.pkl')
trainloss_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_trainloss.pkl')
trainacc_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_trainacc.pkl')
testacc_file = os.path.join(os.getcwd(), 'dataset', f'twolayer_testacc.pkl')


print(f'Save Pickle({train_loss_file}) file....')
with open(params_file, 'wb') as f_params,\
        open(trainloss_file, 'wb') as f_trainloss,\
        open(trainacc_file, 'wb') as f_trainacc,\
        open(testacc_file, 'wb') as f_testacc:
    pickle.dump(network.params, f_params, -1)
    pickle.dump(train_losses, f_trainloss, -1)
    pickle.dump(train_accuracies, f_trainacc, -1)
    pickle.dump(test_accuracies, f_testacc, -1)
print('done!')