import sys
sys.path.append("..")
import numpy as np
from dezero import Variable

def rosenblock(x0, x1):
  y = 100 * (x1 - x0**2) ** 2 + (1 - x0) **2
  return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000

for i in range(iters):
  print(x0, x1)
  
  y = rosenblock(x0, x1)
  
  x0.cleargrad()
  x1.cleargrad()
  y.backward()
  
  x0.data -= lr * x0.grad
  x1.data -= lr * x1.grad