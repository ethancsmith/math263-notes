---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Lecture 7: Systems of first order ODE's.

## Vector-valued functions and their derivatives.

+++

Consider the system of first order ordinary differential equations
```{math}
\frac{\mathrm{d}x}{\mathrm{d}t} &= x + z,\\
\frac{\mathrm{d}y}{\mathrm{d}t} &= x + y,\\
\frac{\mathrm{d}z}{\mathrm{d}t} &= -2x - z.
```

```{code-cell} ipython3
import sympy as sp

# solve the IVP symbolically with the sympy library
t = sp.Symbol('t');
x = sp.Function('x');
y = sp.Function('y');
z = sp.Function('z');
ode = [sp.Eq(x(t).diff(t), x(t) + z(t)), 
       sp.Eq(y(t).diff(t), x(t) + y(t)), 
       sp.Eq(z(t).diff(t), -2*x(t) - z(t))];
soln=sp.dsolve(ode,[x(t), y(t), z(t)], ics={x(0): 1, y(0): sp.Rational(-1,2), z(0): -1}); 
display(soln[0])
display(soln[1])
display(soln[2])
r = sp.Function('r');
r = sp.Matrix([soln[0].rhs, soln[1].rhs, soln[2].rhs]);
r.diff(t)

#rhs=f(x,y(x));
#display(Markdown(f"The true solution to the ODE $y'={sp.latex(rhs)}$ with initial condition $y({a})={y0}$ is ${sp.latex(soln)}$."))

```

```{code-cell} ipython3

```

## Euler's method for vector-valued functions

```{code-cell} ipython3
import numpy as np
import math263

pi = np.pi;

# define IVP parameters
f = lambda x, y: np.array([y[0] + y[2], y[0] + y[1], -2*y[0] - y[2]]);
a, b = 1, 2*pi;
y0=np.array([1, -1/2, -1]);

n = 10;
(xi, y_euler) = math263.euler(f, a, b, y0, n); 
y_euler
```

```{code-cell} ipython3

```
