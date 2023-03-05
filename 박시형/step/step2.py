import numpy as np

from step1 import Variable

class Function:
  def __call__(self, input: Variable):
    x = input.data # 데이터 꺼내기
    y = self.forward(x)
    output = Variable(y)
    return output
  
  def forward(self, x):
    raise NotImplementedError
  

class Square(Function):
  def forward(self, x):
    return x ** 2
  

if __name__ == "__main__":
  x = Variable(np.array(10))
  f = Square()
  y = f(x)

  print(type(y))
  print(y.data)
