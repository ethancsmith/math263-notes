---
jupytext:
  custom_cell_magics: kql
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: math263-notes
  language: python
  name: python3
---

# 4: Runge–Kutta methods.

The **Runge–Kutta methods** are a family of numerical methods which generalize both Euler's method {eq}`euler-method` and Heun's modified Euler method {eq}`mem-method`.
Indeed, the Runge-Kutta method of order $1$ is Euler's method,
while Heun's modified Euler method is an example of a second order Runge–Kutta method.
The essence of the Runge-Kutta methods is to approximate the true solution using additional terms of the Taylor series without actually computing the higher order derivatives.

+++

## Rethinking Heun's modified Euler method.

Previously, we derived Heun's modified Euler method based on numerical integration via the trapezoidal rule.
We now take a different viewpoint that leads to the development of a whole family of second order methods.

Suppose as usual that we are given an IVP of the form
```{math}
y' &= f(x,y),\\
y(x_0) &= y_0,
```
and our goal is to approximate $y(x_1)$, where $x_1 = x_0+h$.
The Taylor series approach previously developed tells us that to first order we can do no better than Euler's method, viz,
```{math}
y(x_1) \approx y(x_0) + y'(x_0)(x_1-x_0) = y_0 + f(x_0,y_0)h.
```
The Taylor series approach also makes it clear that to obtain a second order approximation, we require an estimate for $y''(x_0)$.
Now numerical methods are mostly just ways of faking limits.
Indeed, if $\alpha$ is constant with repect to $h$, then under reasonable assumptions on $f$ and the solution $y$, we have
```{math}
y''(x_0) &= \lim_{h\to 0}\frac{y'(x_0+h) - y'(x_0)}{h}\\
&=\frac{y'(x_0+\alpha h) - y'(x_0)}{\alpha h} + O(h)\\
&=\frac{k_1-k_2}{\alpha h} + O(h)
```
as $h\to 0$, where $k_1 = f(x_0, y_0)$ and $k_2 = f(x_0+\alpha h, y_0 + \alpha hk_1)$.
This estimate then yields second order approximation for $y(x_1)$, namely,
```{math}
y(x_1) &= y(x_0) + y'(x_0)h +\frac{y''(x_0)}{2}h^2 + O(h^3)\\
&= y_0 + k_1h  + \frac{k_2-k_1}{2\alpha h}h^2 + O(h^3)
```
as $h\to 0$.
The **generic Runge–Kutta method of order 2** is therefore defined by the recurrence
```{math}
:label: generic-RK2
y_{i+1} = y_i + h\left(w_1k_1 + w_2k_2\right),
```
where
```{math}
k_1 &= f(x_i, y_i),\\
k_2 &= f(x_i + \alpha h, y_i + h\alpha k_1),\\
w_1 &= 1-\frac{1}{2\alpha},\\
w_2 &= \frac{1}{2\alpha},
```
and $\alpha$ is any positive constant.
We mention the three most common choices here.
Setting $\alpha=1$ yields Heun's (modified Euler) method which we have already met. 
**Ralston's order 2 method**, which also occasionally goes by the name "Heun's method," arises by setting $\alpha=2/3$.
Finally, the **midpoint method** (or **corrected Euler method**) is defined by the choice $\alpha=1/2$.

+++

## Higher-order methods.

It is natural to speculate that higher-order methods could be obtained by taking a weighted average of additional (approximate) slopes "sampled" from the subinterval $[x_i, x_{i+1}]$.
An $s$-stage Runge–Kutta method is the result of an average of $s$ such samples.
The general form of an $s$-stage method is
```{math}
:label: s-stage-RK
y_{i+1} = y_i + h\sum_{j=1}^s b_jk_j,
```
where
```{math}
k_1 &= f(x_i, y_i),\\
k_2 &= f(x_i + c_2h, y_i + ha_{2,1}k_1),\\
&\vdots\\
k_s &= f\big(x_i + c_sh, y_i + h(a_{s,1}k_1 + \dots + a_{s,s-1}k_s)\big).
```
Of course care must be taken when choosing the coefficients so that the method is consistent with the Taylor series of the true solution.

Perhaps the most popular Runge–Kutta method is the **classical fourth order Runge–Kutta method** (**RK4**), which is defined by the recurrence
\begin{equation}
y_{i+1} = \frac{h}{6}\big(k_1 + 2k_2 + 2k_3 + k_4\big),
\end{equation}
where
\begin{align}
k_1 &= f(x_i, y_i),\\
k_2 &= f\big(x_i + h/2, y_i + hk_1/2\big),\\
k_3 &= f\big(x_i + h/2, y_i + hk_2/2\big),\\
k_4 &= f(x_i + h, y_i + hk_3).
\end{align}
As the name would seem to imply, RK4 is an order $4$ numerical method.

+++

## Example.

Below we compare Euler's method with the classical Runge–Kutta method of order 4 (RK4) for the IVP 
\begin{align}
y'&=(y/x)-(y/x)^2,\\
y(1)&=1.
\end{align}

```{code-cell} ipython3
import math263
import numpy as np
import sympy
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.style.use('dark_background');

# define IVP parameters
f = lambda x, y: (y/x) - (y/x)**2;
a, b = 1, 2;
y0=1;

# solve the IVP symbolically with the sympy library
x = sympy.Symbol('x');
y = sympy.Function('y');
ode = sympy.Eq(y(x).diff(x), f(x,y(x)));
soln = sympy.dsolve(ode, y(x), ics={y(a): y0}); 
print("The exact symbolic solution to the IVP");
display(ode);
print(f"with initial condition y({a}) = {y0} is");
display(soln);

# convert the symbolic solution to a Python function and plot it with matplotlib.pyplot
sym_y=sympy.lambdify(x, soln.rhs, modules=['numpy']); 
xvals = np.linspace(a, b, num=100);
fig, ax = plt.subplots(layout='constrained');
ax.plot(xvals, sym_y(xvals), color='b', label=f"${sympy.latex(soln)}$");
ax.set_title(f"$y' = {sympy.latex(f(x,y(x)))}$, $y({a})={y0}$");
ax.set_xlabel(r"$x$");
ax.set_ylabel(r"$y$");
ax.legend([f"${sympy.latex(soln)}$"], loc='upper right');
plt.grid(True)

# numerically solve the IVP with n=10 steps of forward Euler and n=10 steps of RK4
n = 10;
(xi, y_euler) = math263.euler(f, a, b, y0, n);
(xi, y_rk4)   = math263.rk4(f, a, b, y0, n);

# plot numerical solutions on top of true solution
ax.plot(xi, y_euler[:, 0], 'ro:', label="Euler");
ax.plot(xi, y_rk4[:, 0], 'go:', label="RK4");
ax.legend(loc='upper left');
plt.show();

# tabulate the results
print("Global errors for Euler's method and RK4.")
table = np.c_[xi, abs(sym_y(xi) - y_euler[:, 0]), abs(sym_y(xi) - y_rk4[:, 0])];
hdrs = ["i", "x_i", "e_{i,Euler} = |y(x_i)-y_i|", "e_{i,RK4} = |y(x_i)-y_i|"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt=['0.0f', '0.1f', '0.5e', '0.5e'], showindex=True))
```
