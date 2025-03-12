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

# 7: Systems of first order ODE's.

+++

## Systems of equations in vector form.

By a **system of $m$ first order ordinary differential equations** we mean a system of equations of the form
```{math}
y_1' &= f_1(t, y_1, y_2, \dots, y_m),\\
y_2' &= f_2(t, y_1, y_2, \dots, y_m),\\
&\quad \vdots\\
y_m' &= f_m(t, y_1, y_2, \dots, y_m).
```
Since humans are much better at holding onto to one object at time, it is convenient (and conceptually more illuminating) to place the $m$ objects into a single vector equation of the form
```{math}
:label: first-order-vector-ode
\mathbf{y}' = \mathbf{f}(t, \mathbf{y}),
```
where $\mathbf{y}(t) = \langle y_1(t), y_2(t), \dots, y_m(t)\rangle$ is a vector-valued function of the scalar $t$, 
```{math}
:label: vector-derivative
\mathbf{y}' %= \frac{\mathrm{d}\mathbf{y}}{\mathrm{d} t}
=\lim_{h\to 0}\frac{\mathbf{y}(t+h) - \mathbf{y}(t)}{h}
= \begin{pmatrix}
\displaystyle\lim_{h\to 0}\frac{y_1(t+h) - y_1(t)}{h}\\
\displaystyle\lim_{h\to 0}\frac{y_2(t+h) - y_2(t)}{h}\\
\vdots\\
\displaystyle\lim_{h\to 0}\frac{y_m(t+h) - y_m(t)}{h}
\end{pmatrix}
=\begin{pmatrix}
y_1'\\
y_2'\\
\vdots\\
y_m'
\end{pmatrix},
```
and 
\begin{equation*}
\mathbf{f}(t, \mathbf{y}) = 
\begin{pmatrix}
f_1(t, y_1, y_2, \dots, y_m)\\
f_2(t, y_1, y_2, \dots, y_m)\\
\vdots\\
f_m(t, y_1, y_2, \dots, y_m)
\end{pmatrix}.
\end{equation*}
Note that we write our vectors as columns when displayed, but use the angle-bracket row notation (common in multivariable calculus texts) as a spacing-saving device for in-line presentation.

As with scalar ODE's, the solution to a vector ODE is not typically unique.  To specify a unique solution it is sufficient to specify $m$ _initial conditions_ of the form $\mathbf{y}(0) = \mathbf{y}_0 = \langle y_{1,0}, y_{2,0}, \dots, y_{m,0}\rangle$.

+++

## Numerical solutions to systems of ODE's.

Evidently, the derivative {eq}`vector-derivative` of a vector-valued function is the vector of derivatives of the scalar-valued components.  This is true, and even important, but it is much more important that the derivative is defined as a limit in precisely the same way as for the scalar theory.  That means that much of what we already know about the scalar theory continues to work for vectors.  In particular, linearization looks exactly the same.  To see this, we just note that if $h$ is small, then
```{math}
:label: approx-vector-derivative
\mathbf{y}'(t) \approx \frac{\mathbf{y}(t+h) - \mathbf{y}(t)}{h}.
```
Now, solving {eq}`approx-vector-derivative` for $\mathbf{y}(t+h)$ yields the usual linear approximation
\begin{equation*}
\mathbf{y}(t+h) \approx \mathbf{y}(t) + h\mathbf{y}'(t).
\end{equation*}
Since linearization works — and looks exactly the same, Euler's method works — and looks exactly the same.  In particular, to numerically approximate a solution to the first order IVP
```{math}
:label: first-order-vector-ivp
\mathbf{y}' &= \mathbf{f}(t, \mathbf{y})\\
\mathbf{y}(t_0) &= \mathbf{y}_0
```
we have the (vectorized) **forward Euler method**
```{math}
:label: vector-forward-euler
\mathbf{y}_{i+1} = \mathbf{y}_i + h \mathbf{f}(t_i, \mathbf{y}_i).
```
For those less familiar with vector arithmetic (or those interested in implementing the method in a low-level programming language) we note that the above vector formula is equivalent to the $m$ scalar formulas
\begin{align*}
y_{1, i+1} &= y_{1,i} + h f_1(t_i, y_{1, i}, y_{2, i}, \dots, y_{m, i}),\\
y_{2, i+1} &= y_{2,i} + h f_2(t_i, y_{1, i}, y_{2, i}, \dots, y_{m, i}),\\
&\quad \vdots\\
y_{m, i+1} &= y_{m,i} + h f_m(t_i, y_{1, i}, y_{2, i}, \dots, y_{m, i}).
\end{align*}

Fortunately, many higher-level software packages (e.g., NumPy) have built-in routines and overloaded operators for manipulating vectors as though they were scalars.  Since we were careful with our numerical method implementations, this means that our code already works for systems of ODE's.  All we need to do is feed the routines appropriately shaped arrays.

+++

## Measuring errors with vector norms.

To understand errors in higher-dimensional spaces it is helpful to establish the concept of a vector norm.

```{prf:definition}
Suppose that $\boldsymbol v = \langle x_1,\dots, x_n\rangle$ is a vector in $n$-dimensional (real or complex) space.
The **$p$-norm** of $\boldsymbol v$ is nonnegative real number
\begin{equation*}
|\boldsymbol v|_p = \begin{cases}
\displaystyle\left(\sum_{i=1}^n |x_i|^p\right)^{1/p} & \text{if } p\in (0,\infty),\\
\max\{|x_i| : 1\le i\le n\} & \text{if } p = \infty.
\end{cases}
\end{equation*}
```

Vector norms provide a sense of distance in vector spaces that is somewhat akin to a ruler.
In particular, we define the $p$-distance between two vectors, say $\boldsymbol v$ and $\boldsymbol w$, to be the real scalar $|\boldsymbol v - \boldsymbol w|_p$.
Since there are infinitely many choices of $p$, there are infinitely many choices of "ruler."
Deciding which choice makes the most sense depends on the application.
For example, if $\boldsymbol v$ is a "position vector" in ordinary 3-dimensional space, then the choice $p=2$ might make sense as it corresponds to the usual euclidean distance.
However, there are many situations where the vector $\boldsymbol v$ is just an "artificial" construct holding a bunch of disparate scalars together for easy manipulation.
In such cases, it probably makes more sense to measure distances with the $\infty$-norm.

```{prf:definition}
Let $p\in (0,\infty]$, and let $\boldsymbol v$, $\boldsymbol w$ be vector-valued functions of the scalar $t$.
We say that $\boldsymbol v$ is **at most the order of $\boldsymbol w$ as $t$ tends toward $a$**, 
and we write $\boldsymbol v(t) = O\big(\boldsymbol w(t)\big)$ as $t\to a$,
provided that there exist constants $c,\delta>0$ so that
\begin{equation*}
|\boldsymbol v(t)|_p \le c|\boldsymbol w(t)|_p
\end{equation*}
for all $t$ so that $0<|t-a|<\delta$.
```

+++

## Example.

Consider the system of first order ordinary differential equations
```{math}
\frac{\mathrm{d}x}{\mathrm{d}t} &= x + z,\\
\frac{\mathrm{d}y}{\mathrm{d}t} &= x + y,\\
\frac{\mathrm{d}z}{\mathrm{d}t} &= -2x - z
```
together with the initial conditions $x(0) = 1$, $y(0) = -1/2$, and $z(0) = -1$.
Writing $\mathbf r = \langle x, y, z\rangle$ for the solution, we may repackage the system in vector form, writing
```{math}
\frac{\mathrm{d}\mathbf{r}}{\mathrm{d} t} 
&= \begin{pmatrix} x + z\\ x + y\\ -2x - z \end{pmatrix},\\
\mathbf{r}(0) & = \begin{pmatrix} 1\\ -1/2\\ 1\end{pmatrix}.
```
First we solve the system symbolically with SymPy.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import sympy
from tabulate import tabulate

import math263

# solve the IVP symbolically with the sympy library
t = sympy.Symbol("t")
x = sympy.Function("x")
y = sympy.Function("y")
z = sympy.Function("z")
ode_rhs = [x(t) + z(t), x(t) + y(t), -2 * x(t) - z(t)]
ode = [
    sympy.Eq(x(t).diff(t), ode_rhs[0]),
    sympy.Eq(y(t).diff(t), ode_rhs[1]),
    sympy.Eq(z(t).diff(t), ode_rhs[2]),
]
a, b = 0, 2 * pi
soln = sympy.dsolve(
    ode, [x(t), y(t), z(t)], ics={x(a): 1, y(a): sympy.Rational(-1, 2), z(a): -1}
)
soln_rhs = sympy.Matrix([eq.rhs for eq in soln])

print("The exact symbolic solution to the IVP is")
display(soln[0])
display(soln[1])
display(soln[2])
display(soln)
```

Next we plot the solution in $xyz$-space.

```{code-cell}
plt.style.use("dark_background")

# lambdify the symbolic solution
sym_x = sympy.lambdify(t, soln[0].rhs, modules=["numpy"])
sym_y = sympy.lambdify(t, soln[1].rhs, modules=["numpy"])
sym_z = sympy.lambdify(t, soln[2].rhs, modules=["numpy"])
tvals = np.linspace(a, b, num=40)

# plot symbolic solution with matplotlib.pyplot
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
fig.set_size_inches(8, 8)
ax.plot(
    sym_x(tvals),
    sym_y(tvals),
    sym_z(tvals),
    "b",
    label=r"$\mathbf{r}(t)=\langle x(t), y(t), z(t)\rangle$",
)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_title(
    r"$\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} = \langle$"
    f"${sympy.latex(ode_rhs[0])}$, ${sympy.latex(ode_rhs[1])}$, ${sympy.latex(ode_rhs[2])}$"
    r"$\rangle$, $\mathbf{r}(0) = \langle 1, -1/2, -1\rangle$"
)
ax.set_box_aspect(aspect=None, zoom=0.85)
ax.legend(loc="upper right")
ax.grid(True)
```

Now we compute a numerical solution with RK4 and plot it along with the symbolic solution.

```{code-cell}
# define IVP parameters
f = lambda t, r: np.array([r[0] + r[2], r[0] + r[1], -2 * r[0] - r[2]])
a, b = 0, 2 * pi
r0 = np.array([1, -1 / 2, -1])

n = 10
(ti, r_rk4) = math263.rk4(f, a, b, r0, n)

plt.figure(fig)
ax.plot(r_rk4[:, 0], r_rk4[:, 1], r_rk4[:, 2], "ro:", label="RK4")
ax.legend(loc="upper right")
plt.show()
```

Finally, we compute the absolute and relative errors at each mesh point in $t$-space using both the $2$-norm and the $\infty$-norm.

```{code-cell}
rvals = np.c_[sym_x(ti), sym_y(ti), sym_z(ti)]
error_vecs = rvals - r_rk4
p = 2
errors_2 = np.linalg.norm(error_vecs, axis=1, ord=p)
errors_inf = np.linalg.norm(error_vecs, axis=1, ord=np.inf)
table = np.c_[ti, errors_2, errors_inf]
print(f"Global {p}-norm and infinity-norm errors for RK4 solution.")
hdrs = ["i", "t_i", "|r(t_i) - r_i|_2", "|r(t_i) - r_i|_inf"]
print(tabulate(table, hdrs, tablefmt="mixed_grid", 
    floatfmt=["0.0f", "0.5f", "0.5e", "0.5e"], showindex=True))
```
