# 신경망학습: 오차제곱합 손실함수(Sum of Squares Error, SSE)
import os
import sys
import numpy as np
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sum_squares_error
except ImportError:
    print('Library Module Can Not Found')

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])  # 정답 레이블. 답은 2이다.

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.])  # y1, y2, y3는 각각 총합이 1이다.
y2 = np.array([0.1, 0.05, 0.05, 0.6, 0.02, 0.03, 0.05, 0.1, 0., 0.])  # 각각의 index는 확률을 뜻한다.
y3 = np.array([0., 0., 0.95, 0.02, 0.01, 0.01, 0., 0.1, 0., 0.])  # 2일 확률이 95%라는 말.

# test
print(sum_squares_error(y1, t))
print(sum_squares_error(y2, t))
print(sum_squares_error(y3, t))
# 오차가 가장 큰 것은 y2, 가장 적은 것은 y3로 y3가 잘 학습되었다고 할 수 있다.
