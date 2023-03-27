import numpy as np
import weakref
import contextlib

class Variable:
    __array_priority__=200
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0

    def set_creator(self, func):
        self.creator=func
        self.generation=func.generation+1

    def backward(self, retain_grad=False):
        if self.grad is None:#At first, dy/dy(grad) is 1. 
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_set=set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
        add_func(self.creator)
        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs=(gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad=None#y is weak reference
    
    def cleargrad(self):
        self.grad=None

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'variable('+p+')'

    def __mul__(self, other):
        return mul(self, other)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype



class Function:
    def __call__(self, *inputs):
        inputs=[as_variable(x) for x in inputs]

        xs=[x.data for x in inputs]#list comprehension
        ys=self.forward(*xs)#list unpack
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation=max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs=inputs#store input that was used in propagation(need when calculating gradient while doing backpropagation)
            self.outputs=[weakref.ref(output) for output in outputs]
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
    
class Mul(Function):
    def forward(self, x0, x1):
        y=x0*x1
        return y
    
    def backward(self, gy):
        x0,x1=self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0, x1):
    x1=as_array(x1)
    return Add()(x0,x1)

def mul(x0, x1):
    x1=as_array(x1)
    return Mul()(x0, x1)

Variable.__mul__=mul
Variable.__add__=add
Variable.__rmul__=mul
Variable.__radd__=add

def numerical_diff(f, x, eps=1e-4):#eps=h
    x0=Variable(x.data-eps)#(x-h)
    x1=Variable(x.data+eps)#(x+h)
    y0=f(x0)#f(x-h)
    y1=f(x1)#f(x+h)
    return (y1.data-y0.data)/(2*eps)

class Config:
    enable_backprop=True

@contextlib.contextmanager
def using_config(name, value):
    old_value=getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

x=Variable(np.array(2.0))
y=x+np.array(3.0)
print(y)

x=Variable(np.array(2.0))
y=x+3.0
print(y)

x=Variable(np.array(2.0))
y=3.0*x+1.0
print(y)