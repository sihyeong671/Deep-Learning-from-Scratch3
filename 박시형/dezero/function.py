import numpy as np
from dezero import utils
from dezero.core import (
  Function,
  as_variable
)

class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis
    self.keepdims = keepdims
  
  def forward(self, x):
    self.x_shape = x.shape
    y = x.sum(axis=self.axis, keepdims=self.keepdims)
    return y
  
  def backward(self, gy):
    gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) # 수정 필요
    gx = broadcast_to(gy, self.x_shape)
    return gx 
  
def sum(x, axis=None, keepdims=False):
  return Sum(axis, keepdims)(x)

class Sin(Function):
  def forward(self, x):
    y = np.sin(x)
    return y
  def backward(self, gy):
    x, = self.inputs
    gx = gy * cos(x)
    return gx

def sin(x):
  return Sin()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)

class Cos(Function):
  def forward(self, x):
    y = np.cos(x)
    return y
  def backward(self, gy):
    x, = self.inputs
    gx = gy * -sin(x)
    return gx

def cos(x):
  return Cos()(x)

class Tanh(Function):
  def forward(self, x):
    y = np.tanh(x)
    return y
  def backward(self, gy):
    y = self.outputs[0]()
    gx = gy * (1 - y ** 2)
    return gx
  
def tanh(x):
  return Tanh()(x)
  
  
class Reshape(Function):
  def __init__(self, shape) -> None:
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    y = x.reshape(self.shape)
    return y
  
  def backward(self, gy):
    return reshape(gy, self.x_shape)
  
def reshape(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return Reshape(shape)(x)


class Transpose(Function):
  
  def __init__(self, axes=None):
        self.axes = axes
        
  def forward(self, x):
    y = x.transpose(self.axes)
    return y

  def backward(self, gy):
    if self.axes is None:
      return transpose(gy)

    axes_len = len(self.axes)
    inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
    return transpose(gy, inv_axes)

def transpose(x, axes=None):
  return Transpose(axes)(x)


class BroadcastTo(Function):
  def __init__(self, shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    y = np.broadcast_to(x, self.shape)
    return y
  
  def backward(self, gy):
    gx = sum_to(gy, self.x_shape)
    return gx

def broadcast_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)


class SumTo(Function):
  def __init__(self, shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    y = utils.sum_to(x, self.shape)
    return y
  
  def backward(self, gy):
    gx = broadcast_to(gy, self.x_shape)
    return gx

def sum_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)

class MatMul(Function):
  def forward(self, x, W):
    y = x.dot(W)
    return y
  
  def backward(self, gy):
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW

def matmul(x, W):
  return MatMul()(x, W)


class MSE(Function):
  def forward(self, x0, x1):
    diff = x0 - x1
    y = (diff ** 2).sum() / len(diff)
    return y
  
  def backward(self, gy):
    x0, x1 = self.inputs
    diff = x0 - x1
    gx0 = gy * diff * (2. / len(diff))
    gx1 = -gx0
    return gx0, gx1
  

def mean_squared_error(x0, x1):
  return MSE()(x0, x1)


def linear_simple(x, W, b=None):
  t = matmul(x, W)
  if b is None:
    return t
  
  y = t + b
  t.data = None
  return y


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def sigmoid_simple(x):
  x = as_variable(x)
  y = 1 / (1 + exp(-x))
  return y


class Sigmoid(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)
  

def softmax_simple(x, axis=1):
  x = as_variable(x)
  y = exp(x)
  sum_y = sum(y, axis=axis, keepdims=True)
  return y / sum_y


class Softmax(Function):
  def __init__(self, axis=1):
    self.axis = axis
  
  def forward(self, x):
    y = x - x.max(axis=self.axis, keepdims=True)
    y = exp(y)
    y /= y.sum(axis=self.axis, keepdims=True)
    return y
  
  def backward(self, gy):
    y = self.outputs[0]()
    gx = y * gy
    sumdx = gx.sum(axis=self.axis, keepdims=True)
    gx -= y * sumdx
    return gx
    
    
def softmax(x, axis=1):
    return Softmax(axis)(x)
  
    
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        # xp = cuda.get_array_module(t.data)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        # Variable로 변경
        y = (y - t_onehot) * gy
        return y
      
def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)