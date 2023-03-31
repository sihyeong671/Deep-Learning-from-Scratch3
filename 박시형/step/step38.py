import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.function as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.reshape((6,))
y.backward(retain_grad=True)
print(x.grad)