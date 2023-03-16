import numpy as np
from utils import *


class Add(Function):
  def forward(self, x0, x1):
    y = x0 + x1
    return y
  
  def backward(self, gys):
    pass
  
  
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)
