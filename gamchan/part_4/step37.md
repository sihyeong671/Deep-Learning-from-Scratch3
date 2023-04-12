# $x$
|        |        |        |
|--------|--------|--------|
|$x_{11}$|$x_{12}$|$x_{13}$|
|$x_{21}$|$x_{22}$|$x_{23}$|
<br>

***
# $c$
|        |        |        |
|--------|--------|--------|
|$c_{11}$|$c_{12}$|$c_{13}$|
|$c_{21}$|$c_{22}$|$c_{23}$|
<br>


***
# $t$
|        |        |        |
|--------|--------|--------|
|$t_{11} = x_{11} + c_{11}$|$t_{12} = x_{12} + c_{12}$|$t_{13} = x_{13} + c_{13}$|
|$t_{21} = x_{21} + c_{21}$|$t_{22} = x_{22} + c_{22}$|$t_{23} = x_{23} + c_{23}$|
<br>

***
# $y$
$y = t_{11} + t_{12} + t_{13} + t_{21} + t_{22} + t_{23}$

$y = (x_{11} + c_{11}) + (x_{12} + c_{12}) + (x_{13} + c_{13})$<br>
$ + (x_{21} + c_{21}) + (x_{22} + c_{22}) + (x_{23} + c_{23})$<br>
<br>

***
# $\frac{\partial y}{\partial t}$
## 1. Dimension
$y$: &emsp; $1 \times 1$ (scalar)

$t$: &emsp; $2 \times 3$ (matrix)

$\frac{\partial y}{\partial t}$ &emsp; $(1 \times 1) \times (2 \times 3) = 2 \times 3$ (matirx)
<br>
<br>

## 2. Jacobian
|        |        |        |
|--------|--------|--------|
|$\frac {\partial y} {\partial t_{11}}$|$\frac {\partial y} {\partial t_{12}}$|$\frac {\partial y} {\partial t_{13}}$|
|$\frac {\partial y} {\partial t_{21}}$|$\frac {\partial y} {\partial t_{22}}$|$\frac {\partial y} {\partial t_{23}}$|

|        |        |        |
|--------|--------|--------|
|$1$|$1$|$1$|
|$1$|$1$|$1$|
<br>

***
# $\frac{\partial y}{\partial x}$
## 1. Dimension
$y$: &emsp; $1 \times 1$ (scalar)

$x$: &emsp; $2 \times 3$ (matrix)

$\frac{\partial y}{\partial x}$ &emsp; $(1 \times 1) \times (2 \times 3) = 2 \times 3$ (matirx)
<br>
<br>

## 2. Jacobian
|        |        |        |
|--------|--------|--------|
|$\frac {\partial y} {\partial x_{11}}$|$\frac {\partial y} {\partial x_{12}}$|$\frac {\partial y} {\partial x_{13}}$|
|$\frac {\partial y} {\partial x_{21}}$|$\frac {\partial y} {\partial x_{22}}$|$\frac {\partial y} {\partial x_{23}}$|

|        |        |        |
|--------|--------|--------|
|$1$|$1$|$1$|
|$1$|$1$|$1$|
<br>

***
# $\frac{\partial y}{\partial c}$
## 1. Dimension
$y$: &emsp; $1 \times 1$ (scalar)

$c$: &emsp; $2 \times 3$ (matrix)

$\frac{\partial y}{\partial c}$ &emsp; $(1 \times 1) \times (2 \times 3) = 2 \times 3$ (matirx)
<br>
<br>

## 2. Jacobian
|        |        |        |
|--------|--------|--------|
|$\frac {\partial y} {\partial c_{11}}$|$\frac {\partial y} {\partial c_{12}}$|$\frac {\partial y} {\partial c_{13}}$|
|$\frac {\partial y} {\partial c_{21}}$|$\frac {\partial y} {\partial c_{22}}$|$\frac {\partial y} {\partial c_{23}}$|

|        |        |        |
|--------|--------|--------|
|$1$|$1$|$1$|
|$1$|$1$|$1$|
<br>
