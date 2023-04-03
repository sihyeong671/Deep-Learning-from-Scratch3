if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

x = Variable(np.array(2.0))
x.name = "x"
y = x ** 2
y.name = "y"
y.backward(create_graph=True)
gx = x.grad
gx.name = "dy/dx"
x.cleargrad()
plot_dot_graph(gx, verbose=False, to_file='graphs/step36_1.png')

z = gx**3 + y
z.name = "z"
z.backward()
print(x.grad)
plot_dot_graph(z, verbose=False, to_file='graphs/step36_2.png')