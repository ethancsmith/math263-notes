---
jupytext:
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

# 7: Systems of first order ODE's.

## Systems of equations in vector form.

By a **system of $m$ first order ordinary differential equations** we mean a system of equations of the form
```{math}
y_1' &= f_1(x, y_1, y_2, \dots, y_m),\\
y_2' &= f_2(x, y_1, y_2, \dots, y_m),\\
&\quad \vdots\\
y_m' &= f_m(x, y_1, y_2, \dots, y_m).
```
Since humans are much better at holding onto to one object at time, it is convenient (and conceptually more illuminating) to place the $m$ objects into a single vector equation, namely
```{math}
:label: first-order-vector-ode
\mathbf{y}' = \mathbf{f}(x, \mathbf{y}),
```
where $\mathbf{y}(x) = \langle y_1(x), y_2(x), \dots, y_m(x)\rangle$ is a vector-valued function of the scalar $x$, 
```{math}
:label: vector-derivative
\mathbf{y}' %= \frac{\mathrm{d}\mathbf{y}}{\mathrm{d} x}
=\lim_{h\to 0}\frac{\mathbf{y}(x+h) - \mathbf{y}(x)}{h}
= \begin{pmatrix}
\displaystyle\lim_{h\to 0}\frac{y_1(x+h) - y_1(x)}{h}\\
\displaystyle\lim_{h\to 0}\frac{y_2(x+h) - y_2(x)}{h}\\
\vdots\\
\displaystyle\lim_{h\to 0}\frac{y_m(x+h) - y_m(x)}{h}
\end{pmatrix},
```
and 
\begin{equation*}
\mathbf{f}(x, \mathbf{y}) = 
\begin{pmatrix}
f_1(x, y_1, y_2, \dots, y_m)\\
f_2(x, y_1, y_2, \dots, y_m)\\
\vdots\\
f_m(x, y_1, y_2, \dots, y_m)
\end{pmatrix}.
\end{equation*}
Note that we write our vectors as columns when displayed, but use the angle-bracket row notation (common in multivariable calculus texts) as a spacing-saving device for in-line presentation.

As with scalar ODE's, the solution to a vector ODE is not typically unique.  To specify a unique solution it is sufficient to specify $m$ _initial conditions_ of the form $\mathbf{y}(x_)) = \mathbf{y}_0 = \langle y_{1,0}, y_{2,0}, \dots, y_{m,0}\rangle$.

## Numerical solutions to systems of ODE's.

Evidently, the derivative {eq}`vector-derivative` of a vector-valued function is the vector of derivatives of the scalar-valued components.  This is true, and even important, but it is much more important that the derivative is defined as a limit in precisely the same way as for the scalar theory.  That means that much of what we already know about the scalar theory continues to work for vectors.  In particular, linearization looks exactly the same.  To see this, we just note that if $h$ is small, then
```{math}
:label: approx-vector-derivative
\mathbf{y}'(x) \approx \frac{\mathbf{y}(x+h) - \mathbf{y}(x)}{h}.
```
In fact, there is even a theory of multiple-variable Taylor series to precisely quantify what is meant by this assertion though we will not pursue such a precise statement this time.
Now, solving {eq}`approx-vector-derivative` for $\mathbf{y}(x+h)$ yields the usual linear approximation
\begin{equation*}
\mathbf{y}(x+h) \approx \mathbf{y}(x) + h\mathbf{y}'(x).
\end{equation*}
Since linearization works — and looks exactly the same, Euler's method works — and looks exactly the same.  In particular, to numerically approximate a solution to the first order IVP
```{math}
:label: first-order-vector-ivp
\mathbf{y}' &= \mathbf{f}(x, \mathbf{y})\\
\mathbf{y}(x_0) &= \mathbf{y}_0
```
we have the (vectorized) **forward Euler method**
```{math}
:label: vector-forward-euler
\mathbf{y}_{i+1} = \mathbf{y}_i + h \mathbf{f}(x_i, \mathbf{y}_i).
```
For those less familiar with vector arithmetic (or those interested in implementing the method is a low-level programming language) we note that the above vector formula is equivalent to the $m$ scalar formulas
\begin{align*}
y_{1, i+1} &= y_{1,i} + h f_1(x_i, y_{1, i}, y_{2, i}, \dots, y_{m, i}),\\
y_{2, i+1} &= y_{2,i} + h f_2(x_i, y_{1, i}, y_{2, i}, \dots, y_{m, i}),\\
&\quad \vdots\\
y_{m, i+1} &= y_{m,i} + h f_m(x_i, y_{1, i}, y_{2, i}, \dots, y_{m, i}).
\end{align*}

Fortunately, many higher-level software packages (e.g., NumPy) have built-in routines and overloaded operators for manipulating vectors as though they were scalars.  Since we were careful with our numerical method implementations, this means that our code already works for systems of ODE's.  All we need to do is feed the routines appropriately shaped arrays.

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
First we solve the system symbolic with SymPy.

```{code-cell}
import sympy
import numpy as np
from IPython.display import display, Markdown

# solve the IVP symbolically with the sympy library
t = sympy.Symbol('t');
x = sympy.Function('x');
y = sympy.Function('y');
z = sympy.Function('z');
ode_rhs = [x(t) + z(t), x(t) + y(t), -2*x(t) - z(t)];
ode = [sympy.Eq(x(t).diff(t), ode_rhs[0]), 
       sympy.Eq(y(t).diff(t), ode_rhs[1]), 
       sympy.Eq(z(t).diff(t), ode_rhs[2])];
a, b = 0, 2*np.pi;
soln=sympy.dsolve(ode,[x(t), y(t), z(t)], 
                  ics={x(a): 1, y(a): sympy.Rational(-1,2), z(a): -1}); 
soln_rhs = sympy.Matrix([eq.rhs for eq in soln]);

print("The exact symbolic solution to the IVP is");
display(soln[0]);
display(soln[1]);
display(soln[2]);
```

Next we plot the solution in $xyz$-space.

```{code-cell}
import matplotlib.pyplot as plt

plt.style.use('dark_background');

# lambdify the symbolic solution
sym_x = sympy.lambdify(t, soln[0].rhs, modules=['numpy']);
sym_y = sympy.lambdify(t, soln[1].rhs, modules=['numpy']);
sym_z = sympy.lambdify(t, soln[2].rhs, modules=['numpy']);
tvals = np.linspace(a, b, num=40);

# plot symbolic solution with matplotlib.pyplot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'));
fig.set_size_inches(8, 8);
ax.plot(sym_x(tvals), sym_y(tvals), sym_z(tvals), 
        label=r"$\mathbf{r}(t)=\langle x(t), y(t), z(t)\rangle$");
ax.set_xlabel(r"$x$");
ax.set_ylabel(r"$y$");
ax.set_zlabel(r"$z$");
ax.set_title(r"$\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} = \langle$" 
             f"${sympy.latex(ode_rhs[0])}$, ${sympy.latex(ode_rhs[1])}$, ${sympy.latex(ode_rhs[2])}$"
             r"$\rangle$, $\mathbf{r}(0) = \langle 1, -1/2, -1\rangle$");
ax.set_box_aspect(aspect=None, zoom=0.85)
ax.legend(loc="upper right")
ax.grid(True)
```

Now we compute a numerical solution via Euler's method and plot it along with the symbolic solution.

```{code-cell}
import math263

# define IVP parameters
f = lambda t, r: np.array([r[0] + r[2], r[0] + r[1], -2*r[0] - r[2]]);
a, b = 0, 2*np.pi;
r0=np.array([1, -1/2, -1]);

h=0.1;
n = round((b - a)/h + 0.5);
(ti, r_euler) = math263.euler(f, a, b, r0, n); 

plt.figure(fig);
ax.plot(r_euler[:, 0], r_euler[:, 1], r_euler[:, 2], 'ro:', label="Euler's method");
ax.legend(loc="upper right")
plt.show()
```

Finally, we compute the absolute and relative errors at each mesh point in $t$-space using the $2$-norm.

```{code-cell}
from tabulate import tabulate

rvals = np.c_[sym_x(ti), sym_y(ti), sym_z(ti)];
error_vecs = rvals - r_euler
p=2; # set 'p = inf.inf' to use infinity norm
errors = np.linalg.norm(error_vecs, axis=1, ord=p) 
rel_errors = errors/np.linalg.norm(rvals, axis=1, ord=p);
table = np.c_[ti, errors, rel_errors];
hdrs = ["i", "t_i", "abs. error", "rel. error"];
print(f"Global {p}-norm errors for Euler's method.")
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt='0.5g', showindex=True))
```
