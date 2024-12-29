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

# Lecture 11: Finite difference methods for boundary value problems.

The shooting method replaces the given BVP with a family of IVPs which it solves numerically until it finds one that closely approximates the desired boundary condition(s).  The method of finite differences, on the other hand, imposes the boundary condition(s) exactly and instead approximates the differential equation with "finite differences" which leads to a system of equations that can hopefully be solved by a (numerical) equation solver.

## Forward differences and backward differences.

The same germ that lead to Euler's method gives us the finite difference method, viz., if $h$ is small and nonzero, then
```{math}
:label: difference-quotient
y'(x) \approx \frac{y(x+h) - y(x)}{h}
```
since $y'(x)$ is the limit of the difference quotient as $h\to 0$.  When $h>0$, we refer to this approximation as a **forward difference**, and when $h<0$ we refer to the approximation as a **backward difference**. To quantify the error in {eq}`difference-quotient`, we appeal (once again) to Taylor's theorem.  Assuming that $h>0$ and that the exact solution $y$ has sufficiently many continuous derivatives, Taylor's theorem 
```{math}
:label: forward-taylor
y(x+h) = y(x) + y'(x)h + \frac{y''(x)}{2!}h^2 + \frac{y'''(x)}{3!}h^3 + \dots.
```
A bit of algebra then shows us that the forward difference approximation
```{math}
:label: forward-difference-approx
y'(x) = \frac{y(x+h) - y(x)}{h} + O(h)
```
as $h\to 0$.  Furthermore, replacing $h$ by $-h$ in {eq}`forward-taylor` gives
```{math}
:label: backward-taylor
y(x-h) = y(x) - y'(x)h + \frac{y''(x)}{2!}h^2 - \frac{y'''(x)}{3!}h^3 + \dots.
```
Solving for $y'(x)$ as before, we see that the backwards difference approximation
```{math}
:label: backward-difference-approx
y'(x) = \frac{y(x) - y(x-h)}{h} + O(h)
```
as $h\to 0$.  Thus, we see that both the forward difference and the backward difference approximations are order 1 accurate.

## Central differences and higher-order approximations.

The difference in sign pattern between {eq}`forward-taylor` and {eq}`backward-taylor` has a number of happy consequences.  First, subtracting {eq}`backward-taylor` from {eq}`forward-taylor` annihilates all the even order terms.  In particular, we have
\begin{equation*}
y(x+h) - y(x-h) = 2y'(x)h + 2\frac{y'''(x)}{3!}h^3 + \dots.
\end{equation*}
Solving for $y'(x)$ here gives the order 2 accurate **central difference** approximation
```{math}
:label: central-difference-approx
y'(x) = \frac{y(x+h) - y(x-h)}{2h} + O(h^2)
```
as $h\to 0$.

But there's more!  Adding {eq}`backward-taylor` and {eq}`forward-taylor` annihilates all the odd order terms so that we have
\begin{equation*}
y(x+h) + y(x-h) = 2y(x)h + 2\frac{y''(x)}{2!}h^2 + 2\frac{y^{(4)}(x)}{4!}h^4 + \dots.
\end{equation*}
Solving for $y''(x)$, we obtain the approximation
```{math}
y''(x) = \frac{y(x+h) - 2y(x) + y(x-h)}{h^2} + O(h^2)
```
as $h\to 0$.
One can play similar games (playing different Taylor series expressions off one another) to obtain even higher order approximations to $y'(x)$ and $y''(x)$ as well as approximations approximations for higher-order derivatives as needed.

## Finite differences for BVP's.

TODO.

## Advantages and disadvantages.

In practice, achieving acceptable accuracy with a finite difference method requires a very small step-size $h$ as compared to a shooting method.  The smaller the step-size, the greater the number of variables involved in the system of equations.  The increase in variables puts pressure on both computing time and memory resources.  However, it is important to remember that the shooting method requires the numerical solution to a sequence of IVPs until tolerance is achieved.  Though finite difference methods tend to be more memory intensive than shooting methods, there are a number of factors that go into deciding which is more work intensive for a given problem.

## Example.

TODO.

```{code-cell}

```
