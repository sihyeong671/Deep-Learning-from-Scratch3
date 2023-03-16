from utils import *
from memory_profiler import profile

@profile
def use_bp():
  Config.enable_backprop = True
  x = Variable(np.ones((100, 100, 100)))
  y = square(square(square(x)))

@profile
def no_use_bp():
  Config.enable_backprop = False
  x = Variable(np.ones((100, 100, 100)))
  y = square(square(square(x)))

if __name__ == "__main__":
  use_bp()
  no_use_bp()