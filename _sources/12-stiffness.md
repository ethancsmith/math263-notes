---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: math263-notes
  language: python
  name: python3
---

# 12: Stiffness.

## A badly-behaved example.

Consider the initial value problem
```{math}
:label: stiff-example
\dot{y} &= -200y + 200t + 101,\\
y(0)&= 1.01
```
over the $t$-interval $[0, 1]$.
The problem is easy to solve analytically.

```{code-cell}
import numpy
import sympy
from matplotlib import pyplot
from tabulate import tabulate

import math263

pyplot.style.use("dark_background")

# define IVP parameters
f = lambda t, y: -200 * y + 200 * t + 101
a, b = 0, 1
y0 = 1.01

# solve the IVP symbolically with the sympy library
y = sympy.Function("y")
t = sympy.Symbol("t")
ode = sympy.Eq(y(t).diff(t), f(t, y(t)))
soln = sympy.dsolve(ode, y(t), ics={y(a): y0})
print("The function")
display(soln)
print("is the exact symbolic solution to the IVP.")

# convert the symbolic solution to a Python function and plot it with matplotlib.pyplot
sym_y = sympy.lambdify(t, soln.rhs, modules=["numpy"])
tvals = numpy.linspace(a, b, num=300)
fig, ax = pyplot.subplots(layout="constrained")
ax.plot(tvals, sym_y(tvals), linewidth=2.5, label=f"${sympy.latex(soln)}$")
ax.legend(loc="upper left")
ax.set_title(f"${sympy.latex(ode)}$, $y({a})={y0}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$y$")
ax.grid(True)
pyplot.show()
```

Since the solution $y$ is (for the most part) slowly varying over the $t$-interval $[0, 1]$, it seems that this problem _should_ pose no problem for our numerical methods.  However, the  Euler method with step-size $h=0.1$ is awesomely terrible, and RK4 is even worse!

```{code-cell}
# numerically solve the IVP with forward Euler and RK4
h = 0.1
n = round((b - a) / h)
ti, y_euler = math263.euler(f, a, b, y0, n)
ti, y_rk4 = math263.rk4(f, a, b, y0, n)

# tabulate errors
print("Global errors for Euler's method and RK4.")
table = numpy.c_[ti, abs(sym_y(ti) - y_euler[:, 0]), abs(sym_y(ti) - y_rk4[:, 0])]
hdrs = ["i", "t_i", "e_{i,Euler} = |y(t_i)-y_i|", "e_{i,RK4} = |y(t_i)-y_i|"]
print(
    tabulate(
        table,
        hdrs,
        tablefmt="mixed_grid",
        floatfmt=["0.0f", "0.2f", "0.5e", "0.5e"],
        numalign="right",
        showindex=True,
    )
)
```

## Stiffness.

The reason that the above example behaves so poorly is that the desired solution is surrounded by other rapidly decaying transient solutions to the ODE.  The fact that these other solutions converge to our desired solution as $t\to\infty$ should help dampen out the errors in our numerical methods, but the problem here is that the convergence is (relatively) too fast.  It is as though the equation is too stable causing the numerical method to become unstable.  This condition is known as **stiffness**.  "Physically, stiffness corresponds to a process whose components have highly disparate time scales or a process whose time scale is very short compared to the interval over which it is being studied (Heath. _Scientific Computing_, 2e., 2002)."

Below we give a plot of the desired solution to the IVP {eq}`stiff-example` together with several other solutions to the ODE (with different initial value conditions) on top of a direction field.  Observe that the slopes of these other solutions are very steep at locations very near the desired solution.

```{code-cell}
pyplot.figure(fig)
for i in range(1, len(ti)):
    # solve the IVP with initial condition y(t_i) = y_i (from Euler solution)
    solni = sympy.dsolve(ode, y(t), ics={y(ti[i]): y_euler[i, 0]})
    sym_yi = sympy.lambdify(t, solni.rhs, modules=["numpy"])
    tvals = numpy.linspace(ti[i], b, num=300)
    ax.plot(tvals, sym_yi(tvals), linewidth=2.5, label=f"${sympy.latex(solni)}$")

# create direction field for ODE
tmin, tmax = a, b
ymin, ymax = 1 / 2, 3 / 2

# set step sizes defining the horizontal/vertical distances between mesh points
ht, hy = (b - a) / n, 0.1

# sample x- and y-intervals at appropriate step sizes; explicitly creating array of doubles
tvals = numpy.arange(tmin, tmax + ht, ht, dtype=numpy.double)
yvals = numpy.arange(ymin, ymax + hy, hy, dtype=numpy.double)

# create rectangle mesh in xy-plane;
T, Y = numpy.meshgrid(tvals, yvals)
dt = numpy.ones(T.shape)
# create a dx=1 at each point of the 2D mesh
dy = f(T, Y)
# sample dy =(dy/dt)*dt, where dt=1 at each point of the 2D mesh
# normalize each vector <dt, dy> so that it has "unit" length
[dt, dy] = [dt, dy] / numpy.sqrt(dt**2 + dy**2)

# plot direction field on top of previous plot
pyplot.quiver(
    T, Y, dt, dy, color="w", headlength=0, headwidth=1, pivot="mid", label="_nolegend_"
)

pyplot.ylim(ymin, ymax)
pyplot.show()
```

If we plot the first 4 points of the Euler method solution together with each corresponding tangent line, we can see why this situation would tend to "confuse" Euler's method (or really any explicit method).  An explicit method would require a very short step-size $h$ in order to avoid drastically over-shooting its target.

```{code-cell}
B = 4
tvals = numpy.linspace(a, b, n)
ymin = min(y_euler[:B, 0])
ymax = max(y_euler[:B, 0])

# create new figure
fig, ax = pyplot.subplots(layout="constrained")
ax.plot(ti[:B], y_euler[:B, 0], "r:o", label="Euler method")
for i in range(B):
    # solve the IVP with initial condition y(t_i) = y_i (from Euler solution)
    solni = sympy.dsolve(ode, y(t), ics={y(ti[i]): y_euler[i, 0]})
    sym_yi = sympy.lambdify(t, solni.rhs, modules=["numpy"])
    tvals = numpy.linspace(a, b, num=300)
    ax.plot(tvals, sym_yi(tvals), label="_nolegend_")
ax.plot(tvals, sym_y(tvals), "b", linewidth=2.5, label="Exact solution")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$y$")
ax.set_title("(Forward) Euler solution with transients")
pyplot.ylim(ymin - 100, ymax + 100)
ax.legend()
pyplot.show()
```

For this reason, stiff differential equations are typically solved using implicit methods, which tend to be much less sensitive to small perturbations in initial conditions.  The simplest implicit method is the **backward Euler method (BEM)**, which is defined by the equation {eq}`backward-euler`.  By their very nature, implicit methods require an equation solving subroutine.  Typically, Newton's method or some variant is used.  For our Python implementation, we decided to use the `fsolve` routine from the [SciPy](https://scipy.org/) library.

```python
import numpy as np
import scipy as sp

def bem(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a)=y0
    over the interval [a, b] via n steps of the backward Euler method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    for i in range(n):
        func = lambda Y: Y - (y[i] + h * f(x[i + 1], Y))
        y[i + 1] = sp.optimize.fsolve(func, y[i])
    return x, y
```
Below we observe that the backward Euler method has no trouble with the toy problem that has so stymied both the usual (forward) Euler method and RK4.

```{code-cell}
# numerically solve the IVP with forward Euler and RK4
h = 0.1
n = round((b - a) / h)
ti, y_bem = math263.bem(f, a, b, y0, n)

# tabulate errors
print("Backward Euler solution.")
table = numpy.c_[ti, y_bem[:, 0], abs(sym_y(ti) - y_bem[:, 0])]
hdrs = ["i", "t_i", "y_i", "e_{i,BEM} = |y(t_i)-y_i|"]
print(
    tabulate(
        table,
        hdrs,
        tablefmt="mixed_grid",
        floatfmt=["0.0f", "0.2f", "0.5f", "0.5e"],
        numalign="right",
        showindex=True,
    )
)
```

## Off-the-shelf solutions.

The SciPy library implements several IVP solvers.  The `Radau` and `BDF` solvers implement implicit methods that are suitable for stiff equations while `LSODA` automatically detects stiffness and switches between an explicit method and an implicit method as needed.  See [`scipy.integrate`](https://docs.scipy.org/doc/scipy/reference/integrate.html#solving-initial-value-problems-for-ode-systems) for more details.
