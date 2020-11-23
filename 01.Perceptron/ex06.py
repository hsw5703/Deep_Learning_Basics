# step 함수

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd(), 'lib')))
    from common import step

except ImportError:
    print('Library Module Can Not Found')

x = np.arange(-10, 10, 0.1)
y = step(x)

fig, ax = plt.subplots()
plt.plot(x, y)
plt.show()