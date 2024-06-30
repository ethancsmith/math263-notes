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

# Lecture 6: Predictor–corrector methods.

## Predictor–corrector methods.

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
However, there are many important application where $f(x,y)$ is nonlinear in $y$. 
Thus, in order to use an implicit formula such as {eq}`backward-euler`, it is necessary to incorporate some nonlinear equation solver such as Newton's method or fixed-point iteration to solve for $y_{i+1}$.

Since fixed-point iteration techniques require an _initial guess_, it is common to use an explicit method to "predict" the value of $y_{i+1}$ before employing the implicit method to "correct" the guess. 
Thus, the pairing of an explicit predictor method with an implicit corrector method is often referred to as a **predictor-corrector method**. 
The question of how many times to apply the fixed-point iteration of the corrector method can be a delicate matter. 
Each iteration brings us closer to the true solution but at the cost of more evaluations of the function $f(x,y)$. 
In theory, one might choose to iterate the corrector until the value $y_{i+1}$ converges to within some tolerance. 
Since iteration of corrector method such as {eq}`backward-euler` only brings us closer to the solution of {eq}`backward-euler` and not necessarily the true solution $y(x_{i+1})$, it is often not worth the additional cost to apply more than 1 iteration. 
If higher accuracy is needed, it is probably more efficient to reduce the step size $h$.

## Advantages and disadvantages.

The main advantage of a predictor-corrector method is that the implicit (corrector) method tends to give the method better stability properties.  A numerical method is said to be stable if small perturbations in the initial data do not cause the resulting numerical solution to diverge away from the original as x-> \[Infinity].  The obvious disadvantage of a predictor-corrector method is that each iteration the implicit method costs additional functional evaluations which translates to more work.  To mitigate this effect, the number of corrector iterations per step is generally kept low.


```{code-cell} ipython3

```
