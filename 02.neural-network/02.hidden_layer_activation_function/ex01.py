# sigmoid function & graph
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib')) # lib 폴더가 같은 폴더 내에 있지 않고 부모 폴더 아래에 있으므로 .parent를 해줘야 한다.
    from common import sigmoid

except ImportError:
    print('Library Module Can Not Found')

x = np.arange(-10, 10, 0.1)
y = sigmoid(x)

fig, ax = plt.subplots()
plt.plot(x, y)
plt.show()