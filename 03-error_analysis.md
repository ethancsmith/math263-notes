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

# 3: Error analysis.

## Types of error.

In general, numerical methods suffer from two distinct sources of error.
1. **Rounding error** is error introduced due to the use of finite precision arithmetic.
1. **Truncation error** (or **discretization error**) is the error introduced by explicit approximations made in the method.

Although it is important, we will largely ignore rounding error in this course.
Instead we will focus on understanding truncation error, which would remain present even if all arithmetic was performed exactly.
Truncation error can be discussed in two different forms.
The **local truncation error** is the error made in just one step of a numerical method.
In particular, we assume that the previous computed value $y_{i}$ is exactly correct, and then we compute the error at the next step, i.e., the local truncation error in the $(i+1)$st step is the quantity
\begin{equation*}
\ell_{i+1} = y_*(x_{i+1})- y_{i+1},
\end{equation*}
where $y_*$ is the true solution to the ODE that passes through the point $(x_{i}, y_{i})$, but not necessarily to the given IVP.  Put another way, the local truncation error is the extent to which the true solution fails to satisfy the formula the method uses to produce the $(i+1)$st approximation.
The **global truncation error** in the $(i+1)$st step is the difference between the the true solution value and the computed solution, i.e., it is the quantity 
\begin{equation*}
e_{i+1} = y(x_{i+1}) - y_{i+1},
\end{equation*}
where $y$ is the true solution to the given IVP.

When solving an IVP, the global truncation error and the local truncation error always agree at the first step.
Although it is potentially misleading, the global truncation error may be viewed as the accumulation of local errors made at all previous steps.

## The order of a numerical method.

Suppose $f$ and $g$ are functions of a real variable $x$ and that both are defined on an open interval containing $x=a$ except possibly at $x=a$ itself.

```{prf:definition}

We say that $f$ **is at most the of order $g$ as $x$ tends toward $a$**, and we write
\begin{equation*}
f(x) = O\big(g(x)\big)\text{ as } x\to a
\end{equation*}
if there are constants $M, \delta>0$ so that
\begin{equation*}
|f(x)|\le M|g(x)|
\end{equation*}
for all $x$ so that $0<|x-a|<\delta$.
```

For any reasonable numerical method, both the local and the global truncation errors should tend to zero as the step size $h = x_{i+1}-x_i$ tends to zero.
So, we are usually interested in bounding errors as functions of $h$ which are tending toward zero as $h\to 0$.  

```{prf:definition}

We say that a numerical method has **order $p$** if the global trunctation error is $O\big(h^p\big)$ as $h\to 0$.
```

Note that since $h$ is tending toward zero, larger values of $p$ are preferred since higher degree monomials vanish faster near the origin than lower degree monomials do.
For example, $h^3$ vanishes more quickly than $h^2$ which vanishes more quickly than $h$.  

```{note}
In many situations, if the local truncation error $\ell_{i+1} = O\big(h^{p+1}\big)$ as $h\to 0$, then the global truncation error $e_{i+1} = O\big(h^p\big)$ as $h\to 0$.
```

To see this, we argue _heuristically_ as follows.
Suppose that we are numerically approximating the solution to an IVP on the interval $[a,b]$ with step-size $h=(b-a)/n$ and that the local truncation error $\ell_{i+1} = O\big(h^{p+1}\big)$ as $h\to 0$.
Then the global error $e_{n} = y(b) - y_n$ arises from the accumulation of $n = (b-a)/h$ local errors.
Whence
\begin{equation*}
e_n = y(b) -y_n = \frac{b-a}{h}O\big(h^{p+1}\big) = O\big(h^p\big) \text{ as }h\to 0.
\end{equation*}
Note well that this is only a heuristic argument and not a proof.  Why?


In practice, local truncation error is easier to determine analytically than global truncation error, 
and so we will provide analytic "evidence" for the order of numerical methods based on local truncation error.
However, global truncation error is usually easier to measure empirically for test problems where the exact symbolic solution is known.
We will therefore compute global errors for our examples.

## Analysis of Euler's method.

Recall that if $y$ has $p$ derivatives at $x_0$, then the **Taylor polynomial of order $p$ about $x=x_0$ for $y$** is
\begin{equation*}
\sum_{j=0}^p \frac{y^{(j)}(x_0)}{j!}(x-x_0)^j = y(x_0) + y'(x_0)(x-x_0) + \frac{y''(x_0)}{2}(x-x_0)^2 + \dots + \frac{y^{(p)}}{p!}(x-x_0)^p.
\end{equation*}
For $x$ near $x_0$, the Taylor polynomial of order $p$ is the best polynomial approximator of $y$ with degree less than or equal to $p$.
The error in the approximation is usually quantified using Lagrange's remainder theorem.
The following fact is a simplified version of Lagrange's theorem that is convenient for our purposes.

```{prf:theorem} Lagrange's remainder theorem
If $y$ is $(p+1)$-times continuously differentiable on the interval $[a,b]$, then for any $x_i, x_{i+1}\in [a,b]$, 
\begin{equation*}
y(x_{i+1}) = \sum_{j=0}^p \frac{y^{(j)}(x_i)}{j!}h^j + O\big(h^{p+1}\big)\text{ as }h\to 0,
\end{equation*}
where $h=x_{i+1}-x_i$.
```

We now use this result to analyze Euler's method for solving the IVP
\begin{align*}
y' &= f(x,y),\\
y(a)&=y_0.
\end{align*}

```{prf:theorem} 
The local truncation error for Euler's method is $O(h^2)$ as $h\to 0$.
```

```{prf:proof}

Recall that Euler's method defines a sequence of approximate values by the recursion
\begin{equation*}
y_{i+1} = y_i + hf(x_i, y_i).
\end{equation*}
By definition, the local truncation error at the $(i+1)$st step is
\begin{equation*}
\ell_{i+1} = y_*(x_{i+1})- y_{i+1},
\end{equation*}
where $y_*$ is the solution to the ODE that passes through the point $(x_{i}, y_{i})$.
By Lagrange's remainder theorem,
\begin{equation*}
y_*(x_{i+1}) = y_*(x_i) + y_*'(x_i)h + O\big(h^2\big)\text{ as } h\to 0.
\end{equation*}
Since $y_*$ is the solution to the ODE that passes through the point $(x_{i}, y_{i})$, it follows that
\begin{equation*}
y_*(x_{i+1}) = y_i + f(x_i, y_i) h + O\big(h^2\big)\text{ as } h\to 0.
\end{equation*}
Finally, substituting the first and fourth displayed equations into the second, we see that
\begin{equation*}
\ell_{i+1} = y_i + f(x_i, y_i) h + O\big(h^2\big) - y_i - hf(x_i, y_i) = O\big(h^2\big) \text{ as }h\to 0.
\end{equation*}
```

Since the local truncation error for Euler's method is $O(h^2)$ as $h\to 0$, we expect that the global truncation error for the method is $O(h)$ as $h\to 0$.
As it turns out Euler's method is an order $1$ method, but we will not prove this.
Instead we demonstrate this fact empirically for the IVP
\begin{align*}
y'  &= x+y,\\
y(0)&= 1.
\end{align*}
The code block below computes the absolute errors obtained when approximating the value $y(1)$ by Euler's method for various step-sizes.

```{code-cell}
import numpy as np
import sympy
import math263
from tabulate import tabulate 

# define IVP parameters
f = lambda x, y: x + y;
a, b = 0, 1;
y0=1;

# solve the IVP symbolically
x = sympy.Symbol('x');
y = sympy.Function('y');
ode = sympy.Eq(y(x).diff(x), f(x,y(x)));
soln = sympy.dsolve(ode, y(x), ics={y(a): y0}); 
rhs = f(x,y(x));
sym_y=sympy.lambdify(x, soln.rhs, modules=['numpy']);

# solve the IVP numerically via Euler's method
num = 10;
base = 2;
h_vals = [(b - a)/(base**e) for e in range(num)]
errors = [abs(math263.euler(f, a, b, y0, base**e)[1][:, 0][-1] - sym_y(b)) for e in range(num)];
cutdown = [errors[i+1]/errors[i] for i in range(num-1)] # compare size of error to size at previous step-size
cutdown.insert(0, None)

# tabulate/display errors
table = np.c_[h_vals, errors, cutdown];
hdrs = ["h", f"|y({b})-y_n|", "cutdown"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt='0.5f', showindex=True))
```

Since the absolute error $|y(1) - y_n|$ is roughly cut in half each time that the step-size $h$ is cut in half, we view the table above as evidence that the global truncation error for Euler's method is $O(h)$ as $h\to 0$, i.e., Euler's method is an order $1$ numerical method.

## Numerical integration and Euler's method.

Euler's method for numerically solving the IVP
\begin{align*}
y' &= f(x, y),\\
y(x_0) &= y_0
\end{align*}
is based on the idea that $f\big(x_0, y(x_0)\big)$ is the slope of the tangent line to the graph of $y$ at $x=x_0$, and therefore
\begin{equation*}
\begin{split}
y(x_0+h)&\approx y(x_0) + hf\big(x_0, y(x_0)\big)\\
&= y_0 + hf(x_0, y_0).
\end{split}
\end{equation*}
Many numerical methods for ODE's can also be derived from the perspective of numerical integration.
For this reason, we often speak of "integrating the ODE."
Indeed, if $y$ is the true solution to the IVP above, then the fundamental theorem of calculus tells us that
```{math}
:label: FTC
y(x_0+h)-y_0=y(x_1)-y(x_0)=\int_{x_0}^{x_0+h}y'(x)\mathrm{d}x=\int_{x_0}^{x_0+h}f\big(x,y(x)\big)\mathrm{d} x.
```
The difficulty then is in evaluating the integral whose integrand depends on the unknown function $y(x)$.
Euler's method follows by approximating the integral with the left-hand rule, namely
```{math}
y(x_0+h)=y_0+\int_{x_0}^{x_0+h}f\big(x,y(x)\big)\mathrm{d} x \approx y_0+hf\big(x_0, y(x_0)\big).
```

## Heun's modified Euler method.

We now derive a method of Heun by replacing the left-hand rule with the trapezoidal rule.
Returning to equation {eq}`FTC` and assuming that the expression $f(x,y)$ has a continuous second derivative (as a function of $x$) on the interval $[x_0, x_0+h]$, the trapezoidal rule gives
```{math}
:label: trapezoidal-approx
y(x_0+h)-y_0 
&=\int_{x_0}^{x_0+h}f\big(x,y(x)\big)\mathrm{d} x\\
&=\frac{h}{2}\Big(f\big(x_0, y(x_0)\big)+f\big(x_{0}+h, y(x_0+h)\big)\Big) +O\big(h^3\big)\\
&=\frac{h}{2}\Big(f(x_0, y_0)+f\big(x_{0}+h, y(x_0+h)\big)\Big) +O\big(h^3\big)
```
as $h\to 0$.
Unfortunately, we do not know how to evaluate $f\big(x_0+h,y(x_0+h)\big)$ since it depends on $y(x_0+h)$ and our goal is to approximate $y(x_0+h)$.
Our solution is to approximate the $y(x_0+h)$ appearing on the right-hand side of {eq}`trapezoidal-approx` by the original Euler method.
In particular,
\begin{equation*}
y(x_0+h) = y_0 + hf(x_0, y_0) + O\big(h^2\big)\text{ as } h\to 0.
\end{equation*}
Substituting this estimate into {eq}`trapezoidal-approx`, we can use Taylor series to argue that
\begin{equation*}
y(x_0+h)-y_0=\frac{h}{2}\Big(f(x_0, y_0)+f\big(x_0+h,y_0+hf(x_0,y_0)\big)\Big)+O\big(h^3\big)\text{ as }h\to 0.
\end{equation*}

The **modified Euler method** (**MEM**) (also known as **Heun's method**) is then defined by the recurrence
```{math}
:label: mem-method
y_{i+1} = y_i + \frac{h}{2}\left(k_1 + k_2\right),
```
where
```{math}
k_1 &= f(x_i, y_i),\\
k_2 &= f(x_i+h, y_i+hk_1).
```
The above derivation demonstrates the following result, which we take as evidence for the fact that the modified Euler method is an order $2$ numerical method.

```{prf:theorem}

The local truncation error for the modified Euler's method is $O(h^3)$ as $h\to 0$.
```
