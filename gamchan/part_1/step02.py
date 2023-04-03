import numpy as np

class Variable:
    def __init__(self, data):
        self.data=data

data=np.array(1.0)
x=Variable(data)
print(x.data)

x.data=np.array(2.0)
print(x.data)

class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2
    
x=Variable(np.array(10))
f=Square()
y=f(x)

print(type(y))
print(y.data)