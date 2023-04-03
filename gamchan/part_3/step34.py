if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph

x = Variable(np.linspace(-7, 7, 200))
x.name = "(-7, 7)"
y = F.sin(x)
labels = ["y=sin(x)", "y'", "y''", "y'''"]
y.name = labels[0]
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    gx.name = "d^{}y/dx^{}".format(i+1, i+1)
    y.name = labels[i+1]
    x.cleargrad()
    gx.backward(create_graph=True)
    plot_dot_graph(gx, verbose=False, to_file='graphs/step34_{}.png'.format(i))
    
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()