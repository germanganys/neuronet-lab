from collections import deque
import numpy as np

class LiveLFilter:
    def __init__(self, b, a):
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b)) 
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)

    def __call__(self, x):
        self._xs.appendleft(x)
        y = (np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)) / self.a[0]
        self._ys.appendleft(y)
        return y