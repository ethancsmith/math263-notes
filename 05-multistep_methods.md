---
jupytext:
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

# Lecture 5: Multistep methods.

The methods that we have considered so far are called **single-step methods** because the point $(x_{i+1}, y_{i+1})$ is computed using only information from the previous step, namely $(x_i, y_i)$.
In particular, the methods have all depended on a formula of the form 
\begin{equation*}
y_{i+1} = y_i + h\phi(x_i, y_i, h).
\end{equation*}
These methods are also called **starting methods** because the initial-value condition $y(x_0)=y_0$ gives enough information to use these methods from the start.
On the other hand, **multistep methods** use more than one of the previously computed $y_j$.
An **$m$-step method** would use the previous $m$ values: $(x_i,y_i), (x_{i-1},y_{i-1}),\dots, (x_{i-m+1},y_{i-m+1})$.
A **linear $m$-step method** takes the form
```{math}
:label: linear-multistep-form
y_{i+1} = \sum_{j=1}^m\alpha_jy_{i+1-j} + h\sum_{j=0}^m\beta_jf(x_{i+1-j}, y_{i+1-j}).
```
Clearly such a method cannot be used until $m$ previous values are known.
For this reason, these methods must be paired with a starting method and are themselves called **continuing methods**.
However, there is additional wrinkle if $\beta_0\ne 0$ since in that case $y_{i+1}$ appears on both sides of the defining equation.
If $\beta_0=0$, the method is said to be **explicit**.
Otherwise, the method is said to be **implicit**.
If the method is implicit and the function $f(x, y)$ is linear in $y$, then it is easy to solve for $y_{i+1}$ once the specific IVP is given, but in practice this is often not the case.
We will consider implicit methods in the next lecture.
For now, we concentrate on the case of explicit methods.
Some of the most popular explicit linear multistep methods are the _Adams–Bashforth methods_.

+++

## Adams–Bashforth 2-step method.

For an explicit, linear $2$-step method equation {eq}`linear-multistep-form` simplifies to
```{math}
:label: 2-step-form
y_{i+1} = \alpha_1y_i + h\big(\beta_1f(x_i,y_i) + \beta_2f(x_{i-1},y_{i-1}\big).
```
The idea of the Adams–Bashforth method is to force this formula to be exact for the first 3 terms of the Taylor series for the true solution $y$.
This will ensure that the local truncation error for the method is $O(h^3)$ as $h\to 0$.
Note that with only $3$ parameters to determine (viz, $\alpha_1, \beta_1, \beta_2)$, this is the best that can be done.
By linearity, it is enough to make it exact for the cases $y(x)=1$, $y(x)=x$, and $y(x)=x^2$.
If $y(x)=1$, then $f(x,y(x)) = y'(x) = 0$, and so the equation {eq}`2-step-form` becomes
```{math}
:label: ab2-condition-1
1=\alpha_1.
```
Similarly, if $y(x)=x$, then $y'(x)=1$ and {eq}`2-step-form` becomes
```{math}
:label: ab2-condition-2
x_{i+1}=\alpha_1 x_i + h(\beta_1+\beta_2).
```
Finally, if $y(x)=x^2$, then $y'(x)=2x$ and {eq}`2-step-form` becomes
```{math}
:label: ab2-condition-3
x_{i+1}^2 = \alpha_1x_i^2+ h(2\beta_1x_i + 2\beta_2x_{i-1}).
```
Now, equations {eq}`ab2-condition-1`, {eq}`ab2-condition-2`, and {eq}`ab2-condition-3` must hold for all values of $x_i$ and $h$.
Choosing $x_{i}=1$ and $h=1$ so that $x_{i-1}=0$ and $x_{i+1}=2$, we arrive at the system of equations
\begin{align*}
1&=\alpha_1,\\
2&=\alpha_1+\beta_1+\beta_2,\\
4&=\alpha_1+2\beta_1.
\end{align*}
Solving the system yields $\alpha_1=1$, $\beta_1=3/2$, and $\beta_2=-1/2$.
The **Adams–Bashforth two-step method (AB2)** is therefore defined by the recurrence
\begin{equation*}
y_{i+1} = y_i + \frac{h}{2}\big(3y_i' - y_{i-1}'\big),
\end{equation*}
where $y_i' = f(x_i, y_i)$ for each $i\ge 1$.

As we noted above, the local truncation error for the method is $O(h^3)$ as $h\to 0$.
This makes AB2 an order $2$ method.
Thus, it should be paired with a starter method whose order of accuracy is at least $2$ such as Heun's modified Euler method.

+++

## Adams–Bashforth 4-step method.

In a similar manner, one may derive the **Adams–Bashforth four-step method (AB4)**, which is defined by the recurrence
\begin{equation*}
y_{i+1}=y_i+\frac{h}{24}\big(55y_i'-59y_{i-1}'+37y_{i-2}'-9y_{i-3}'\big)
\end{equation*}
and has local truncation error that is $O(h^5)$ as $h\to 0$.
Thus, it should be paired with an order $4$ starting method such as the classical RK4 method.

+++

## Advantages and disadvantages.

The disadvantage of a multistep method is that it is typically slightly more difficult to implement than a single-step method.
It even requires a single-step method to get started.
However, there tends to be a major gain in order of accuracy per work.
We usually measure the work per step of a numerical method based on the number of times that we need to evaluate the "right-hand side" function $f(x,y)$.
The typical single-step Runge-Kutta method of order 2 (e.g., Heun's modified Euler method) requires 2 functional evaluations per step.
We compare this to the Adams–Bashforth 2-step method which has the same order of accuracy and only requires one additional functional evaluation per step since it saves the evaluations from previous steps until it is finished with them.
Furthermore, since the error $|y(x_i)-y_i|$ tends to increase with $i$, it is reasonable to hope that a multistep may have better accuracy (not just order of accuracy) by mere fact that it explicitly incorporates more accurate previous data when approximating $y_{i+1}$.

+++

## Example.

We now conduct an empirical comparison of AB2 with Heun's modified Euler method (MEM) using the IVP
\begin{align*}
y'&=y-x^2+1,\\
y(0)&=1/2
\end{align*}
as a test case.
First we use both methods to solve the problem over the interval $[0,2]$ in $n=10$ steps.
We then present the global truncation errors $e_i=|y(x_i) - y_i|$ for each method at each step in a table.

```{code-cell}
import math263
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import display, Markdown

# define IVP parameters
f = lambda x, y: y - x**2 + 1;
a, b = 0, 2;
y0 = 1/2;

# solve the IVP symbolically with the sympy library
x = sp.Symbol('x');
y = sp.Function('y');
ode = sp.Eq(y(x).diff(x), f(x,y(x)));
soln = sp.dsolve(ode, y(x), ics={y(a): y0}); 
#rhs=f(x,y(x));
#display(Markdown(f"The true solution to the ODE $y'={sp.latex(rhs)}$ with initial condition $y({a})={y0}$ is ${sp.latex(soln)}$."))
sym_y = sp.lambdify(x, soln.rhs, modules=['numpy']);

# numerically solve the IVP with n=10 steps of forward Euler and n=10 steps of RK4
n = 10;
(x, y_mem) = math263.mem(f, a, b, y0, n);
(x, y_ab2) = math263.ab2(f, a, b, y0, n); 

# tabulate the results
print(f"Comparison of global errors for MEM and AB2\
 across interval for step-size h = {(b - a)/n}.")
table = np.transpose(np.stack((x, abs(sym_y(x)-y_mem[:, 0]), abs(sym_y(x)-y_ab2[:, 0]))));
hdrs = ["i", "x_i", "MEM global error", "AB2 global error"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt='0.5f', showindex=True))
```

We now compare the global error for AB2 at the right endpoint of the interval with that of MEM as we shrink the step-size.

```{code-cell}
# compute abs errors at right endpoint for various step-sizes
base = 10;
max_exp = 7;
num_steps = [base**j for j in range(1, max_exp)];
h = [(b-a)/n for n in num_steps];
mem_errors = [abs(math263.mem(f, a, b, y0, n)[1][:, 0][-1]-sym_y(b)) for n in num_steps];
ab2_errors = [abs(math263.ab2(f, a, b, y0, n)[1][:, 0][-1]-sym_y(b)) for n in num_steps];

# tabulate the results
print(f"Comparison of global errors |y_n - y({b})| for various step-sizes.")
table = np.transpose(np.stack((h, mem_errors, ab2_errors)));
hdrs = ["step-size", "MEM global error", "AB2 global error"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt=['0.7f','g','g']))
```

We also compare the empirical average running time for our implementations as the number of steps increases.

```{code-cell}
import timeit
num_trials = 10;
mem_times = [timeit.timeit(lambda: math263.mem(f, a, b, y0, base**j), number=num_trials)/num_trials 
             for j in range(1, max_exp)];
ab2_times = [timeit.timeit(lambda: math263.ab2(f, a, b, y0, base**j), number=num_trials)/num_trials 
             for j in range(1, max_exp)];

# tabulate the results
print(f"Comparison of global errors |y_n - y({b})| for various step-sizes.")
table = np.transpose(np.stack((num_steps, mem_times, ab2_times)));
hdrs = ["num steps", "MEM time (secs)", "AB2 time (secs)"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt=['0.0f', '0.5f', '0.5f']))
```

## Exercises.

1. Implement the Adams–Bashforth four-step method (AB4) with the classical Runge-Kutta method (RK4) as starter.
1. Solve each of the following IVP's with the indicated parameter choices using both AB4 and (single-step) RK4.
For each IVP, create a single table displaying the exact value $y(x_i)$, the approximate value $y_i$ for each method, and the absolute error $|y_i-y(x_i)|$ for each method.
Comment on which method appears to perform better for each problem.
     1. $y'=x\mathrm{e}^{3x}-2y$, $y(0)=0$, $[a,b]=[0,1]$, $h=0.1$.
     1. $y'=1+(x-y)^2$, $y(2)=1$, $[a,b]=[2,3]$, $h=0.1$.
     1. $y'=1+y/x$, $y(1)=2$, $[a,b]=[1,2]$, $h=0.1$.
     1. $y'=\cos 2x - \sin 3x$, $y(0)=1$, $[a,b]=[0,1]$, $h=0.1$.
1. Consider the IVP $y'=1+y/x+(y/x)^2, y(1)=0$.
     1. Use both AB4 and (single-step) RK4 to approximate $y(3)$ for $n=2^k$ with $k=3,\dots, 10$.
     Display the absolute errors for both in a single table.
     Do the data appear consistent with the "advertised" local truncation error for AB4?  Explain.
     1. Record the time that each method requires to perform each of the above calculations and present the results together in a table for easy comparison.
     Comment on what you observe.
