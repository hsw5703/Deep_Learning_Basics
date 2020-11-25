# Broadcasting

import numpy as np

a = np.array([[1, 2, 3, 4, 5],  # (3 * 5) matrix
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]])
b = np.array([1, 2, 3, 4, 5])  # 5 vector
d = a + b  # 5 vector가 3개 행으로 늘어나며 행렬 합 연산된다.
print(d)

# (3 * 5) matrix에 (3 * 1) matrix를 더하면 (3 * 1) matrix가 늘어나 (3 * 5)가 되어 연산되지만
# 대신 3 vector의 경우에는 Error가 발생한다. vector의 경우에는 matrix의 열의 수에 맞춰야 한다.
