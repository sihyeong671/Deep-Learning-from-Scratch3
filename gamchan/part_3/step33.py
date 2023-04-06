if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def f(x):
    y = x**4 - 2 * x**2
    return y

x = Variable(np.array(2.0))
x.name = 'x'
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    y.name = "y=f({})".format(x.data)
    x.cleargrad()
    y.backward(create_graph=True)
    
    gx = x.grad
    gx.name = "gx"+str(i)
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data
    plot_dot_graph(gx, verbose=False, to_file='graphs/step33_gx{}.png'.format(i))