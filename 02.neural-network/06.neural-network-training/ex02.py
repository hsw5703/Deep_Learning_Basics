# 신경망학습: 교차 엔트로피 손실함수(Cross Entropy Error, CEE)

import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import cross_entropy_error
except ImportError:
    print('Library Module Can Not Found')

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0., 0.1, 0., 0.])
y2 = np.array([0.1, 0.05, 0.05, 0.6, 0.02, 0.03, 0.05, 0.1, 0., 0.])
y3 = np.array([0., 0., 0.95, 0.02, 0.01, 0.01, 0., 0.1, 0., 0.])

# test
print(cross_entropy_error(y1, t)) # vector가 들어감
print(cross_entropy_error(y2, t))
print(cross_entropy_error(y3, t))
