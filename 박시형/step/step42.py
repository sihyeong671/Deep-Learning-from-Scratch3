import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.function as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
  y = F.matmul(x, W) + b
  return y

def mse(x0, x1):
  diff = x0 - x1
  return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
  y_pred = predict(x)
  loss = mse(y, y_pred)
  W.cleargrad()
  b.cleargrad()
  loss.backward()
  
  W.data -= lr * W.grad.data
  b.data -= lr * b.grad.data
  print(W, b, loss)