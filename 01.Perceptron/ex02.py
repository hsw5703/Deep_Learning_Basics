# AND gate : perceptron
import os
import sys
import numpy as np
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd(), 'lib')))
    from common import step

except ImportError:
    print('Library Module Can Not Found')


def AND(x):
    w, b = np.array([0.5, 0.5]), np.array(-0.7)  # 가중치(w)와 편향(b)를 변화시켜 AND, OR, NAND를 만든다.

    a = np.sum(x * w) + b
    y = step(a)

    return y


if __name__ == '__main__':
    # 다른 소스 파일 등에서 이 소스의 AND() 함수만을 가져가 사용하고 싶다고 할 때 사용하면 된다.
    # 만약 위 구문이 없으면 아래의 식들도 같이 실행되어 원하지 않는 소스의 부분도 같이 가져가게 된다.
    y1 = AND(np.array([0, 0]))
    print(y1)

    y2 = AND(np.array([0, 1]))
    print(y2)

    y3 = AND(np.array([1, 0]))
    print(y3)

    y4 = AND(np.array([1, 1]))
    print(y4)
