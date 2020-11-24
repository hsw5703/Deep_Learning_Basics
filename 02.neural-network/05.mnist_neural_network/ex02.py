# MNIST 손글씨 숫자 분류 신경망 : 신호전달
import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import init_network, load_mnist
except ImportError:
    print('Library Module Can Not Found')

# 1. 매개변수(w, b) 데이터 셋 가져오기
network = init_network()
w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

print(w1.shape) # 784 * 50(은닉 1층) matrix
print(w2.shape) # 50 * 100 matrix
print(w3.shape) # 100 * 10 matrix
print(b1.shape) # 50 vector
print(b2.shape) # 100 vector
print(b3.shape) # 10 vector

(train_x, train_t), (test_x, test_t) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
