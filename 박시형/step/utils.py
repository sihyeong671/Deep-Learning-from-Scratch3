import numpy as np
import weakref
import contextlib


class Config:
  enable_backprop = True
  
@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config("enable_backprop", False)
  

class Variable:
  __array_priority__ = 200
  def __init__(self, data, name=None) -> None:
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError(f'{type(data)}는 지원하지 않는 타입입니다.')
    self.data = data
    self.grad = None
    self.creator = None
    self.name = name
    self.generation = 0
    
  @property
  def shape(self):
    return self.data.shape
  
  @property
  def size(self):
    return self.data.size
  
  @property
  def dtype(self):
    return self.data.dtype
  
  def __len__(self):
    return len(self.data)
  
  def __repr__(self):
    if self.data is None:
      return "Variable(None)"
    else:
      p = str(self.data).replace("\n", "\n" + ' '*9)
      return f"Variable( {p} )"
  
  # __mul__
  # __add__
  # __rmul__
  
  def cleargrad(self):
    self.grad = None
  
  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1
  
  def backward(self, retain_grad=False):
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
    
    while len(funcs) != 0:
      f = funcs.pop()
      gys = [output().grad for output in f.outputs]
      gxs = f.backward(*gys)
      if not isinstance(gxs, tuple):
        gxs = (gxs,)
      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad = gx
        else:
          x.grad = x.grad + gx
        if x.creator is not None:
          add_func(x.creator)
          
      if not retain_grad:
        for y in f.outputs:
          y().grad = None
    
    
class Function:
  def __call__(self, *inputs):
    inputs = [self._as_variable(x) for x in inputs]
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys, tuple):
      ys = (ys,)
    outputs = [Variable(self._as_array(y)) for y in ys]
    
    if Config.enable_backprop:
      self.generation = max([x.generation for x in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs
      self.outputs = [weakref.ref(output) for output in outputs]
    
  

    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, xs):
    raise NotImplementedError

  def backward(self, gys):
    raise NotImplementedError
  
  @staticmethod
  def _as_array(x):
    if np.isscalar(x):
      return np.array(x)
    return x
  
  @staticmethod
  def _as_variable(x):
    if isinstance(x, Variable):
      return x
    return Variable(x)
  
  
class Square(Function):
  def forward(self, x):
    return x **2
  
  def backward(self, gy):
    x = self.inputs[0].data
    gx = 2 * x * gy
    return gx


class Exp(Function):
  def forward(self, x):
    return np.exp(x)
  
  def backward(self, gy):
    x = self.input.data
    gx = np.exp(x) * gy
    return gx
  
  
class Add(Function):
  def forward(self, x0, x1):
    y = x0 + x1
    return y
  
  def backward(self, gy):
    return gy, gy
  
  
def square(x):
  return Square()(x)


def exp(x):
  return Exp()(x)


def add(x0, x1):
  return Add()(x0, x1)


def numerical_diff(f, x: Variable, eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2 * eps)