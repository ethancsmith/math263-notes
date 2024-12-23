---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Lecture 6: Predictor–corrector methods.

## The trouble with implicit methods.

The numerical methods for solving ODE's that we have considered so far have all been explicit, 
which is to say that the formula defining the $(i+1)$st approximation can be explicitly solved for $y_{i+1}$.
Implicit methods are characterized by formula which are not able to solved in the general case.
Suppose as usual that we are trying to numerically solve the first order IVP
\begin{equation*}
y' = f(x,y),\ y(x_0)=y_0.
\end{equation*}
Linearization at the $i$th mesh point $x_i$ and projecting forward along the tangent line yields the (forward) Euler method
\begin{equation*}
y_{i+1} = y_{i} + h f(x_i, y_i),
\end{equation*}
which we have already discussed in detail. 
On the other hand, linearization at the $(i+1)$st mesh point $x_{i+1}$ and projecting backward along the tangent line yields the **backward Euler method**
```{math}
:label: backward-euler
y_{i+1} = y_{i} + h f(x_{i+1}, y_{i+1}).
```

Since the precise form of the "right-hand side" $f(x,y)$ is not known in advance, we cannot explicitly solve this equation for $y_{i+1}$ in the general case – that is without knowing $f(x,y)$. 
If the function $f(x,y)$ turns out to be linear in $y$, then equation {eq}`backward-euler` can easily be solved for $y$. 
However, there are many important applications where $f(x,y)$ is nonlinear in $y$. 
Thus, in order to use an implicit formula such as {eq}`backward-euler`, it is necessary to incorporate some nonlinear equation solver such as Newton's method or fixed-point iteration to solve for $y_{i+1}$.

## Predictor–corrector methods.

Since fixed-point iteration techniques require an _initial guess_, it is common to use an explicit method to "predict" the value of $y_{i+1}$ before employing the implicit method to "correct" the guess. 
Thus, the pairing of an explicit predictor method with an implicit corrector method is often referred to as a **predictor-corrector method**. 
The question of how many times to apply the fixed-point iteration of the corrector method can be a delicate matter. 
Each iteration brings us closer to convergence but at the cost of more evaluations of the function $f(x,y)$. 
In theory, one might choose to iterate the corrector until the value $y_{i+1}$ converges to within some tolerance. 
Since iteration of corrector method such as {eq}`backward-euler` only brings us closer to the solution of {eq}`backward-euler` and not necessarily the true solution $y(x_{i+1})$, it is often not worth the additional cost to apply more than 1 iteration. 
If higher accuracy is needed, it is probably more efficient to reduce the step size $h$.

## Advantages and disadvantages.

The main advantage of a predictor-corrector method is that the implicit (corrector) method tends to give the method better stability properties.  A numerical method is said to be **stable** if small perturbations in the initial data do not cause the resulting numerical solution to diverge away from the original as $x\to\infty$.  The obvious disadvantage of a predictor-corrector method is that each iteration of the implicit method costs additional functional evaluations which translates to more work.  To mitigate this effect, the number of corrector iterations per step is generally kept low.

+++

## Adams–Moulton 1-step (implicit) method.

As usual, consider the first order IVP
\begin{align*}
y' &= f(x, y)\\
y(x_0) &= y_0.
\end{align*}
Choosing a model of the form
\begin{equation*}
y_{i+1} = \alpha_1y_i + h \left(\beta_0y_{i+1}'+ \beta_1y_i'\right),
\end{equation*}
where $y_i'=f(x_i,y_i)$ for each $i\ge 0$, we proceed by the method of underdetermined coefficients.  In particular, we force the model to be exact for the first three monomials  $y(x)=1$, $y(x)=x$, and $y(x)=x^2$.  Thus, by a calculation that is precisely similar to the derivation that we gave for the Adams–Bashforth two-step (explicit) method, we ultimately arrive at the formula for the **Adams–Moulton one-step (implicit) method**, namely, 
```{math}
y_{i+1} = y_i + \frac{h}{2}(y_{i+1}' + y_i').
```
This method is often paired with the order 2 Adams–Bashforth two-step (explicit) method to create the **order 2 Adams–Bashforth–Moulton (predictor-corrector) method (ABM2)**
```{math}
\hat{y_{i+1}}&=y_i+\frac{h}{2}\left(3y_i'-y_{i-1}'\right),\\
y_{i+1}&=y_i + \frac{h}{2}\left(\hat{y_{i+1}}'+ y_i'\right),
```
where $y_i'=f(x_i,y_i)$ and $\hat{y_{i+1}}' = f(x_{i+1}, \hat{y_{i+1}}')$ for each $i\ge 0$.  Here $\hat{y_{i+1}}'$ is referred to as the **predicted value** and $y_{i+1}$ the **corrected value**.

+++

## Adams–Moulton 3-step (implicit) method.

In a similar manner, one may derive the **Adams–Moulton three-step (implicit) method**, which is defined by the formula
```{math}
y_{i+1}=y_i+\frac{h}{24}(9y_{i+1}'+19y_i'-5y_{i-1}'+y_{i-2}')
```
and has local truncation error that is $O(h^5)$.  It is, therefore, considered an order 4 method.
This method is commonly paired with the order 4 Adams--Bashforth four-step (explicit) method to create the **order 4 Adams–Bashforth–Moulton (predictor-corrector) method (ABM4)**
```{math}
\hat{y_{i+1}}&=y_i+\frac{h}{24}(55y_i'-59y_{i-1}'+37y_{i-2}'-9y_{i-3}'),\\
y_{i+1}&=y_i+\frac{h}{24}(9\hat{y_{i+1}}'+19y_i'-5y_{i-1}'+y_{i-2}')
```
where $y_i'=f(x_i,y_i)$ and $\hat{y_{i+1}}' =f(x_{i+1},\hat{y_{i+1}})$ for each $i\ge 0$.

+++

## Example.

We now compare the Adams–Bashforth 2-step method (AB2) and the Adams–Bashforth–Moulton 2-step method (ABM2) for our test IVP 
```{math}
y'&=y-x^2+1,\\ 
y(0)&=1/2
```
over the interval $[a,b]=[0,2]$ with $n=10$ steps.

```{code-cell}
import math263
import numpy as np
import sympy
from tabulate import tabulate

# define IVP parameters
f = lambda x, y: y - x**2 + 1;
a, b = 0, 2;
y0 = 1/2;

# solve the IVP symbolically with the sympy library
x = sympy.Symbol('x');
y = sympy.Function('y');
ode = sympy.Eq(y(x).diff(x), f(x, y(x)));
soln = sympy.dsolve(ode, y(x), ics={y(a): y0}); 
sym_y = sympy.lambdify(x, soln.rhs, modules=['numpy']);

# numerically solve the IVP with n=10 steps of AB2 and n=10 steps of ABM2
n = 10;
(x, y_ab2) = math263.ab2(f, a, b, y0, n); 
(x, y_abm2) = math263.abm2(f, a, b, y0, n);

# tabulate the results
print(f"Comparison of global errors for AB2 and ABM2 across interval for step-size h = {(b - a)/n}.")
table = np.c_[x, abs(sym_y(x)-y_ab2[:, 0]), abs(sym_y(x)-y_abm2[:, 0])];
hdrs = ["i", "x_i", "AB2 global error", "ABM2 global error"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt='0.5f', showindex=True))
```

Below we include a comparison of the global errors at the right-most endpoint of the interval.  It is evident that each is an order 2 method.  However, the ABM2 predictor-corrector method is more accurate for this example.

```{code-cell}
# compute abs errors at right endpoint for various step-sizes
base = 10;
max_exp = 7;
num_steps = [base**j for j in range(1, max_exp)];
h = [(b-a)/n for n in num_steps];
ab2_errors = [abs(math263.ab2(f, a, b, y0, n)[1][:, 0][-1]-sym_y(b)) for n in num_steps];
abm2_errors = [abs(math263.abm2(f, a, b, y0, n)[1][:, 0][-1]-sym_y(b)) for n in num_steps];

# tabulate the results
print(f"Comparison of global errors |y_n - y({b})| for various step-sizes.")
table = np.c_[h, ab2_errors, abm2_errors];
hdrs = ["step-size", "AB2 global error", "ABM2 global error"];
print(tabulate(table, hdrs, tablefmt='mixed_grid', floatfmt=['0.6f','0.6g','0.6g']))
```
