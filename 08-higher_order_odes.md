---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 8: Higher order initial value problems.

An **$m$th order ordinary differential equation** in *normal form* is an equation of the form
```{math}
:label: mth-order-ode
y^{(m)}(x) = f(x, y, y', \dots, y^{(m-1)}).
```
To obtain a unique solution, it is typically necessary to specify $m$ initial conditions, say $y(x_0) = \alpha_0, y'(x_0) = \alpha_1, \dots, y^{(m-1)}(x_0) = \alpha_{m-1}$.  To numerically solve such a problem we introduce $m$ auxiliary variables (really functions) so as to express the problem as a first order system IVP.  Thus, we reduce the problem to one we already know how to solve.  More precisely, we let $u_0=y, u_1=y', \dots, u_{m-1}=y^{(m-1)}$.  Then equation {eq}`mth-order-ode` is equivalent to the system
\begin{align*}
u_0' &= u_1,\\
u_1' &= u_2,\\
&\quad \vdots\\
u_{m-2}' &= u_{m-1},\\
u_{m-1}' &= f(x, u_0, u_1,\dots, u_{m-1}).
\end{align*}
In this way, we are able to compute a numerical solution $y$ the $m$th order IVP along with its lower order derivatives $y', y'', \dots , y^{(m-1)}$.

+++

## Example.

Consider the second-order initial-value problem
```{math}
:label: higher-order-example
y'' - 2y' + 2y &= \exp(2x)\sin x,\\
y(0) &= -2/5,\\
y'(0)&= -3/5.
```

```{code-cell}
import numpy as np
import sympy
import matplotlib.pyplot as plt

plt.style.use('dark_background');

# solve the IVP symbolically with the sympy library
x = sympy.Symbol('x');
y = sympy.Function('y');
ode = sympy.Eq(y(x).diff(x, 2)-2*y(x).diff(x) + 2*y(x), sympy.exp(2*x)*sympy.sin(x));
a, b = 0, 1;
alpha = [sympy.Rational(-2,5), sympy.Rational(-3,5)];
soln = sympy.dsolve(ode, y(x), ics={y(a): alpha[0], y(x).diff(x).subs(x, a): alpha[1]});
soln = sympy.simplify(soln);

print("The exact symbolic solution to the IVP is");
display(soln);
Dsoln_rhs = sympy.simplify(soln.rhs.diff(x));

# lambdify the symbolic solution
sym_y = sympy.lambdify(x, soln.rhs, modules=['numpy']);
sym_Dy = sympy.lambdify(x, Dsoln_rhs, modules=['numpy']);
xvals = np.linspace(a, b, num=40);

# plot symbolic solution with matplotlib.pyplot
fig, axs = plt.subplots(ncols=2, figsize=(8, 5), layout="constrained");
axs[0].plot(xvals, sym_y(xvals), label=f"${sympy.latex(soln)}$");
axs[1].plot(xvals, sym_Dy(xvals), label=f"$y'(x) = {sympy.latex(Dsoln_rhs)}$");
fig.suptitle(f"${sympy.latex(ode)}$, $y({a}) = {alpha[0]}$, $y'({a}) = {alpha[1]}$")
for i in range(len(axs)):
	axs[i].set_xlabel(r"$x$");
	axs[i].legend(loc="upper left");
	axs[i].grid(True)
axs[0].set_ylabel(r"$y$");
axs[1].set_ylabel(r"$y'$");
```

To solve {eq}`higher-order-example` numerically, we introduce the variables $u_0=y$ $u_1=y'$, and we rewrite the IVP as
```{math}
u_0' &= u_1,\\
u_1' &= 2u_1 - 2u_0 + \exp(2x)\sin x,\\
u_0(0) &= -2/5,\\
u_1(0) &= -3/5
```

```{code-cell}
import math263

# define IVP parameters
f = lambda x, u: np.array([u[1], 2*u[1] - 2*u[0] + np.exp(2*x)*np.sin(x)]);
alpha = [-2/5, -3/5];

n = 5;
(xi, u_rk4) = math263.rk4(f, a, b, alpha, n); 

plt.figure(fig);
for i in range(len(axs)):
	axs[i].plot(xi, u_rk4[:, i], 'ro:', label='RK4 solution');
	axs[i].legend(loc="upper left");
plt.show();
```

```{code-cell}
from tabulate import tabulate

uvals = np.c_[sym_y(xi), sym_Dy(xi)];
errors = np.abs(uvals - u_rk4)
table = np.c_[xi, u_rk4[:, 0], errors[:, 0], u_rk4[:, 1], errors[:, 1]];
hdrs = ["i", "x_i", "y_i", "|y(x_i) - y_i|", "y_i'", "|y'(x_i) - y_i'|"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt='0.5g', showindex=True))
```
