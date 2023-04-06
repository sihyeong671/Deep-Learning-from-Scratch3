from numpy.lib.index_tricks import fill_diagonal
from numpy.core.numeric import isscalar
import numpy as np
import unittest
import weakref
import contextlib

class Config: # 역전파 활성화
  enable_backprop = True

@contextlib.contextmanager
def using_config(name,value):
  old_value = getattr(Config,name)
  setattr(Config,name,value)
  try:
    yield
  finally:
    setattr(Config,name,old_value)

def no_grad():
  return using_config('enable_backprop',False)

class Variable:
    def __init__(self,data,name= None):
      if data is None:
        if not isinstance(data,np.ndarray):   # isinstance는 주어진 객체가 특정 클래스의 인스턴스인지 판별할 때 사용
          raise TypeError
      self.data = data    # 변수값 저장
      self.name=name
      self.grad = None    # 미분값 저장
      self.creator = None # 역전파 시에 이전 함수를 저장
      self.generation = 0;

      Variable.__mul__=mul # 연산자 오버로딩
      Variable.__add__=add  
      Variable.__neg__=neg
      Variable.__sub__=sub
      Variable.__rsub__=rsub
      Variable.__truediv__=div
      Variable.__rtruediv__=rdiv
      Variable.__pow__=pow

    @property
    def shape(self):
      return self.data.shape
    @property
    def ndim(self):
      return self.data.ndim
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
        return 'variable(None)'
      p= str(self.data).replace('\n','\n'+''*9)
      return 'variable('+p+')'

    def set_creator(self,func):
      self.creator =func
      self.generation = func.generation +1

    def backward(self,retain_grad=False):
      if self.grad is None:
        self.grad=np.ones_like(self.data) # 아무것도 없을 때 시작 값
      
      funcs = []
      seem_set =set()

      def add_func(f):  # 리스트를 세대 순으로 정렬
        if f not in seem_set:
          funcs.append(f)
          seem_set.add(f)
          funcs.sort(key=lambda x : x.generation) # funcs 리스트 요소 중에서 generation이 작은 거부터 큰 거 순으로 정렬
      add_func(self.creator) 
      
      while funcs:
        f= funcs.pop()
        gys = [output().grad for output in f.outputs] # f.outputs 리스트에 있는 모든 요소의 .grad 속성을 리스트로 추출하여 반환
        gxs = f.backward(*gys)
        if not isinstance(gxs,tuple):
          gxs = (gxs,)

        for x,gx in zip(f.inputs,gxs):  # zip은 데이터를 묶어서 처리하는 함수
          if x.grad is None:
            x.grad = gx
          else :
            x.grad = x.grad +gx

          if x.creator is not None:
            add_func(x.creator)

        if not retain_grad:
          for y in f.outputs:
            y().grad = None
        
        
class Function:
    def __call__(self,*inputs): # *은 받은 값을 튜플 형태로 저장하는 역할
      xs = [x.data for x in inputs]
      ys = self.forward(*xs)
      if not isinstance(ys,tuple): # 입력값이 튜플이 아니면 튜플로 바꿔주기
        ys = (ys,)
      outputs = [Variable(as_variable(y)) for y in ys] # as_array는 주어진 값을 넘파이로 바꿔줌

      if Config.enable_backprop:
        self.generation = max([x.generation for x in inputs]) # inputs의 x.generation 중에서 가장 큰 값을 골라서 저장 
        for output in outputs :
          output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
    
      return outputs if len(outputs)> 1 else outputs[0]

    def forward(self,x):
      raise NotImplementedError() # 아직 구현하지 않은 부분에 대한 표식

    def backward(self,gy):
      raise NotImplementedError()


class Square(Function):
    def forward(self,x):
      y=x**2
      return y

    def backward(self,gy):
      x=self.inputs[0].data
      gx=2*x*gy
      return gx

class Exp(Function):
    def forward(self,x):
      y= np.exp(x)
      return y

    def backward(self,gy):
      x=self.input.data
      gx=np.exp(x)*gy
      return gx

class Add(Function):
  def forward(self,x0,x1):
    y=x0+x1
    return (y,) # 튜플 형태
  
  def backward(self,gy):
    return gy,gy

class Mul(Function):
  def forward(self,x0,x1):
    y=x0*x1
    return y
  def backward(self,gy):
    x0,x1= self.inputs[0].data, self.inputs[1].data
    return gy*x1,gy*x0

class Neg(Function):
    def forward(self,x):
      return -x
    def forward(self,gy):
      return -gy

class Sub(Function):
  def forward(self,x0,x1):
    y= x0-x1
    return y
  def backward(self,gy):
    return gy,-gy

class Div(Function):
  def forward(self,x0,x1):
    y= x0/x1
    return y
  def backward(self,gy):
    x0,x1=self.inputs[0].data,self.inputs[1].data
    gx0=gy/x1
    gx1=gy*(-x0/x1**2)
    return gx0,gx1

class Pow(Function):
  def __init__(self,c):
    self.c=c
  def forward(self,x):
    y=x**self.c
    return y
  def backward(self,gy):
    x= self.inputs[0].data
    c=self.c
    gx=c*x**(c-1)*gy
    return gx

def numerical_diff(f,x,eps=1e-4):
  x0=Variable(x.data-eps)
  x1=Variable(x.data+eps)
  y0=f(x0)
  y1=f(x1)
  return (y1.data-y0.data)/(2*eps)

class SquareTest(unittest.TestCase):
  def Test_forward(self):
    x=Variable(np.array(2.0))
    y=square(x)
    expected = np.array(4.0)
    self.assertEqual(y.data,expected)
  def Test_Backward(self):
    x=Variable(np.array(3.0))
    y=square(x)
    y.backward()
    expected = np.array(6.0)
    self.assertEqual(x.grad,expected)
  def Test_gradient_check(self):
    x=Variable(np.random.rand(1))
    y=square(x)
    y.backward()
    num_grad = numerical_diff(square,x)
    flg=np.allclose(x.grad,num_grad)
    self.assertTrue(flg)

def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

def as_variable(obj):
  if isinstance(obj,Variable):
    return obj
  return Variable(obj)

def square(x):
  f=Square()
  return f(x)

def exp(x):
  f=Exp()
  return f(x)

def add(x0,x1):
  return Add()(x0,x1)
def mul(x0,x1):
  return Mul()(x0,x1)
def neg(x):
  return Neg()(x)
def sub(x0,x1):
  return sub()(x0,x1)
def rsub(x0,x1):
  x1=as_array(x1)
  return Sub()(x1,x0)
def div(x0,x1):
  x1=as_array(x1)
  return Div(x0,x1)
def rdiv(x0,x1):
  x1=as_array(x1)
  return Div()(x1,x0)
def pow(x,c):
  return Pow(c)(x)

with no_grad():
  x= Variable(np.array(2.0))
  y= square(x)
