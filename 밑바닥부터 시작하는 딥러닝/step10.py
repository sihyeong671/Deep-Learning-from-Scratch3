from numpy.lib.index_tricks import fill_diagonal
from numpy.core.numeric import isscalar
import numpy as np
import unittest

class Variable:
    def __init__(self,data):
      if data is None:
        if not isinstance(data,np.ndarray):
          raise TypeError
      self.data = data    # 변수값 저장
      self.grad = None    # 미분값 저장
      self.creator = None # 역전파 시에 이전 함수를 저장

    def set_creator(self,func):
      self.creator =func

    def backward(self):
      if self.grad is None:
        self.grad=np.ones_like(self.data)
      
      funcs = [self.creator]
      while funcs:
        f= funcs.pop()
        x,y = f.input, f.output
        x.grad = f.backward(y.grad)

        if x.creator is not None:
          funcs.append(x.creator)


class Function:
    def __call__(self,input):
      x= input.data
      y= self.forward(x) # 원하는 함수 저장
      output = Variable(as_array(y))
      output.set_creator(self) # 출력 변수의 창조자를 설정
      self.input = input # 입력변수 저장(역전파할 때 쓰임)
      self.output = output # 출력 변수 저장
      return output

    def forward(self,x):
      raise NotImplementedError() # 아직 구현하지 않은 부분에 대한 표식

    def backward(self,gy):
      raise NotImplementedError()


class Square(Function):
    def forward(self,x):
      y=x**2
      return y

    def backward(self,gy):
      x=self.input.data
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

def square(x):
  f=Square()
  return f(x)
def exp(x):
  f=Exp()
  return f(x)
def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

x= Variable(np.array(0.5))
y=square(exp(square(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)
