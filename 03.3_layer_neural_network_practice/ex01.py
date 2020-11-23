# 3층 신경망 신호 전달 구현

import numpy as np

x = np.array([1., 5.])
print(f'x dimension : {x.shape}') # 2 vector

w1 = np.array([[0.1, 0.2, 0.5], [0.3, 0.4, 1.]])
print(f'w1 dimension : {w1.shape}') # 2 * 3 matrix

b1 = np.array([0.1, 0.2, 0.3])
print(f'b1 dimension : {b1.shape}') # 3 vector

# 신경망에 들어오는 순서대로 생각. wx + b (X) xw + b (O)
a1 = np.dot(x, w1) + b1
print(a1)