#!/usr/bin/env python
# coding: utf-8

# ### import

# In[1]:


import numpy as np
import unittest


# Widely Using Function

# In[2]:


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# ### Variable Declaration

# In[3]:


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def cleargrad(self):
        self.grad = None
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
                
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    add_func(x.creator)


# ### Function(Parent class) Declaration

# In[4]:


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()


# In[5]:


class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx


# In[6]:


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx


# In[7]:


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy


# In[8]:


def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


# In[9]:


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)


# ### Checking
# make backprop working well by initializing generation(BFS?)

# In[10]:


generations = [2, 0, 1, 4, 2]
funcs = []

for g in generations:
    f = Function()
    f.generation = g
    funcs.append(f)
    
print([f.generation for f in funcs])


# In[11]:


funcs.sort(key=lambda x: x.generation)
print([f.generation for f in funcs])

f = funcs.pop()
print(f.generation)


# In[12]:


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)

