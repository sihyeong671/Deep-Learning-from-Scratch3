from step01 import Variable
from step02 import Function, Square
from step03 import Exp
import numpy as np

def numerical_diff(f, x, eps=1e-4):#eps=h
    x0=Variable(x.data-eps)#(x-h)
    x1=Variable(x.data+eps)#(x+h)
    y0=f(x0)#f(x-h)
    y1=f(x1)#f(x+h)
    return (y1.data-y0.data)/(2*eps)#central difference formula (f(x+h)-f(x-h))/2h
'''
f=Square()#f(x)=x^2
x=Variable(np.array(2.0))
dy=numerical_diff(f,x)
print(dy)
'''
def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))

x=Variable(np.array(0.5))
dy=numerical_diff(f,x)
#print(dy)
#result: 3.2974426293330694
#=> as x decreases or increases by eps, y changes by 3.297.. times h
#numerical difference has a error, and huge amount of computation => backpropagation