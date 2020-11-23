import numpy as np


# step 활성함수

def step(x):
    return np.array(x > 0, dtype=int)  # True / False로 출력하지 말고 0 / 1로 출력하라는 뜻.