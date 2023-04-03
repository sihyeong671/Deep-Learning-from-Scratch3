if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt

def f(x):
    y=x**4-2*x**2
    return y

def gx2(x):
    return 12*x**2-4

x=Variable(np.array(2.0))
iters=10
#X=[x.data.astype(float)]
#Y=[x.data.astype(float)**4-2*x.data.astype(float)**2]
for i in range(iters):
    print(i,x)
    y=f(x)
    x.cleargrad()
    y.backward()
    x.data-=x.grad/gx2(x.data)
    #Y.append(y.data.astype(float))
    #X.append(x.data.astype(float))
'''
plt.plot(X, Y, 'ro')
x_=np.arange(-3,4)
y_=x_**4-2*x_**2
print(x_)
print(y_)
plt.plot(x_, y_)
plt.show()
'''