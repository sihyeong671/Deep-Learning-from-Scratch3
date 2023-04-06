import sys
sys.path.append("..")
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def f(x):
  y = x ** 4 - 2 * x ** 2
  return y

x = Variable(np.array(2.0))
iters = 10

# result = f(x)
# plot_dot_graph(result, to_file="step33.png")


for i in range(iters):
  print(i, x)

  y = f(x)
  x.cleargrad()
  y.backward(create_graph=True)

  gx = x.grad
  x.cleargrad()
  gx.backward()
  gx2 = x.grad

  x.data -= gx.data / gx2.data

