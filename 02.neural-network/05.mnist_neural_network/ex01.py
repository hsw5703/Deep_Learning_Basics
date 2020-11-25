# MNIST 손글씨 숫자 분류 신경망 : 데이터 살펴보기
import os
import sys
from PIL import Image # python에서 image를 볼 수 있게 해주는 라이브러리
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
except ImportError:
    print('Library Module Can Not Found')

(train_x, train_t), (test_x, test_t) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
# normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
# one_hot_label = True : train_t와 test_t에 해당하는 값에만 적용된다.
# 0~9까지의 숫자가 train_t(학습데이터의 정답데이터)에 담겨있는데 이를 0과 1로 나타낸다.
# 예를 들어 5인 경우, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 로 5번 칸에 1이 나온다.
# 이는 사용자 정의 함수로 mnist.py에 들어가서 살펴보면 알 수 있다.

print(train_x.shape) # 60000 * 784 matrix 학습데이터
print(train_t.shape) # 60000 vector 학습데이터의 정답

t = train_t[0]
print(t) # 5

x = train_x[0]
print(x.shape) # 784 vector(28 * 28)의 이미지 형태를 일렬로 세운 상태로 저장되어 있다.

x = x.reshape(28, 28) # 형상을 원래 이미지 형태(28 * 28)로 바꾼다. 값은 RGB 값으로 되어 있다.
print(x.shape)

# 이미지 보기 : PIL(Python Image Library) 사용

pil_image = Image.fromarray(x)
pil_image.show()