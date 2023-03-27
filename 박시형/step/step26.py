import sys
sys.path.append("..")
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

# x0 = Variable(np.array(1.0))
# x1 = Variable(np.array(1.0))
# y = x0 + x1
# txt = _dot_func(y.creator)
# print(txt)

def goldstein(x, y):
  z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
    (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
  return z

def test(x, y):
  z = x**2 + y**2
  return z

x = Variable(np.array([1.0, 2.1]))
y = Variable(np.array([1.0, 3.0]))
z = goldstein(x, y)
# z = test(x, y)
z.backward()

x.name = "x"
y.name = "y"
z.name = "z"
plot_dot_graph(z, verbose=True, to_file="goldstein.png")