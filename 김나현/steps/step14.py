import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self, func):
        self.creator=func

    def backward(self):
        if self.grad is None:#At first, dy/dy(grad) is 1. 
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        #print(funcs)
        while funcs:
            f=funcs.pop()
            gys=[output.grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs=(gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    funcs.append(x.creator)
    
    def cleargrad(self):
        self.grad=None

class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]#list comprehension
        ys=self.forward(*xs)#list unpack
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs#store input that was used in propagation(need when calculating gradient while doing backpropagation)
        self.outputs=outputs
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):#gy=ndarray->pass gradient from output part
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    
    def backward(self, gy):
        x=self.inputs[0].data
        gx=2*x*gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y=np.exp(x)
        return y
    
    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0, x1):
    return Add()(x0,x1)

def numerical_diff(f, x, eps=1e-4):#eps=h
    x0=Variable(x.data-eps)#(x-h)
    x1=Variable(x.data+eps)#(x+h)
    y0=f(x0)#f(x-h)
    y1=f(x1)#f(x+h)
    return (y1.data-y0.data)/(2*eps)

x=Variable(np.array(3.0))
y=add(x,x)
y.backward()
print(x.grad)

x.cleargrad()
y=add(add(x,x),x)
y.backward()
print(x.grad)
