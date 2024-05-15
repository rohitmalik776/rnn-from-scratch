import numpy as np

a = np.arange(start=0, stop=12).reshape((4, 3))
b = np.ones(shape=(4, 1))

c = a + b
print(c)