import sys
sys.path.append("..")
import numpy as np
from dezero import Variable

print(__file__)

def sphere(x, y):
  z = x**2 + y**2
  return z

def matyas(x, y):
  z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
  return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)
