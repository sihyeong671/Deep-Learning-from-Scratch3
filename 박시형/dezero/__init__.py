is_simple_core = True

if is_simple_core:
  from dezero.core_simple import (
    Variable,
    Function,
    using_config,
    no_grad,
    as_array,
    as_variable,
    setup_variable
  )
else:
  pass

setup_variable()