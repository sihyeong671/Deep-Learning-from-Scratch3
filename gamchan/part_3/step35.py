if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)
iter = 0

for i in range(iter):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    
gx = x.grad
gx.name = "gx0"
plot_dot_graph(gx, verbose=False, to_file='graphs/step35.png')