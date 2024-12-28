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

# Lecture 10: Boundary value problems and the shooting method.

## Boundary value problems.

Side conditions must be imposed in order for a system of ODE's to have a unique a solution.  If all the side conditions are imposed as a single point, then the problem is called an initial value problem (IVP).  If the side conditions are imposed at more than one point, then we have a **boundary value problem (BVP)**.  In typical situations, the conditions are imposed at the endpoints (boundary) of an interval $[a,b]$, which of course explains the name.  

Since higher-order equations can be expressed as systems of first order equations, we can restrict attention to problems of the form
```{math}
:label: first-order-system-ode
\mathbf{y}' = \mathbf{f}(x, \mathbf y)
```
over the interval $a\le x\le b$ together with the condition
```{math}
:label: boundary-conditions
\mathbf{g}(\mathbf{y}(a), \mathbf{y}(b)) = \mathbf{0},
```
where as usual bold symbols denote vectors of the appropriate dimension.

## The shooting method.

The **shooting method** is a trick for turning this type of problem into a "sequence" of problems that we already know how to solve.  The idea is to replace the BVP by an IVP with unknown parameters.  In particular, we ignore any conditions imposed on $\mathbf{y}(b)$ by the boundary condition {eq}`boundary-conditions`.  Then we fix any values of the vector $\mathbf{y}(a)$ that can be determined from the start; the rest we treat as unknowns.  We then "guess" the unknown parameters in $\mathbf{y}(a)$ and solve the resulting IVP.  We then check to see if numerical result satisfies {eq}`boundary-conditions` to within desired tolerance.

## Example.

Consider for example the second-order BVP
```{math}
:label: bvp-example
y''  &= 4y,\\
y(0) &= 0,\\
y(1) &= 5.
```
First we let $u_0 = y$, $u_1 = y'$, and we rewrite the BVP {eq}`bvp-example` as
```{math}
u_0' &= u_1,\\
u_1' &= 4u_0,\\
u_0(0) &= 0,\\
u_0(1) &= 5.
```
Then we collect the new variables into vectors $\mathbf u = \langle u_1, u_2\rangle$, $\mathbf{f}(x, \mathbf u) = \langle u_1, 4u_0\rangle$, and $\mathbf{g}(\mathbf{u}(0), \mathbf{u}(1)) = \langle u_0(0), u_0(1)-5\rangle$, and observe that the problem is now expressed in the form of {eq}`first-order-system-ode` and {eq}`boundary-conditions`.

For the shooting method, we ignore what the boundary condition $\mathbf{g} = \mathbf 0$ has to say about $\mathbf u(1)$.  Instead we introduce another unknown parameter $s$, and we consider the IVP
```{math}
:label: IVP-family
\mathbf{u}' &= \mathbf{f}(x, \mathbf{u}),\\
\mathbf{u}(0) &= \begin{pmatrix} 0\\ s\end{pmatrix}.
```
The goal then is to find a value of $s = u_1(0) = y'(0)$ so that when we solve the resulting IVP $u_0(1)\approx 5$, i.e., $|u_0(1) - 5|$ is within a specified tolerance.  Below we make an initial guess of $s = 1$ and solve the IVP with $n = 10$ steps of RK4.

```{code-cell}
import numpy
import math263
from matplotlib import pyplot

# set up IVP parameters
f = lambda x, u: numpy.array([u[1], 4*u[0]]);
a, b = 0, 1;
bcond = 0, 5; # boundary conditions on y = u_0
u0 = lambda s: numpy.array([0, s]); # set u_1(0) = s unknown

# numerically solve the IVP with n = 10 steps of RK4
n = 10;
s = 1;
xi, ui = math263.rk4(f, a, b, u0(s), n)

print(f"With s = {s}, we have y_{n} = u_{0, n} = {ui[-1, 0]:0.6g}");

fig, ax = pyplot.subplots(layout='constrained');
ax.plot([a, b], bcond, "go");
ax.plot(xi, ui[:, 0], ":.", label=f"RK4 $s$ = {s}")
ax.set_xlabel(r"$x$");
ax.set_ylabel(r"$y$");
ax.legend(loc="upper left");
```

The guess $s=1$ produces an approximate value for $y(1) = u_0(1)$ that is too low.  This suggests that we should try a steeper guess for $s = u_1(0) = y'(0)$.  Below we try again with $s=2$.

```{code-cell}
# numerically solve the IVP with n = 10 steps of RK4
n = 10;
s = 2;
xi, ui = math263.rk4(f, a, b, u0(s), n)

print(f"With s = {s}, we have y_{n} = u_{0, n} = {ui[-1, 0]:0.6g}");

pyplot.figure(fig);
ax.plot(xi, ui[:, 0], ":.", label=f"RK4 $s$ = {s}")
ax.set_xlabel(r"$x$");
ax.set_ylabel(r"$y$");
ax.legend(loc="upper left");
pyplot.show()
```

Again we are too low.  So, we try again with $s=3$.

```{code-cell}
# numerically solve the IVP with n = 10 steps of RK4
n = 10;
s = 3;
xi, ui = math263.rk4(f, a, b, u0(s), n)

print(f"With s = {s}, we have y_{n} = u_{0, n} = {ui[-1, 0]:0.6g}");

pyplot.figure(fig);
ax.plot(xi, ui[:, 0], ":.", label=f"RK4 $s$ = {s}")
ax.set_xlabel(r"$x$");
ax.set_ylabel(r"$y$");
ax.legend(loc="upper left");
pyplot.show()
```

This time we are too high, and that is fortunate for now we have "bracketed" $s$.  In particular, we "know" (assuming that the problem varies continuously with $s$) that $s\in (2,3)$.  We can therefore bisect the interval and guess $s=(2+3)/2=5/2$.  If that guess is too low, we know that $s\in(5/2, 3)$.  If it is too high, we know that $s\in (2, 5/2)$.  We can then keep bisecting until our computed value $u_{0,n} = y_n$ is within some desired tolerance of our target $y(1)=5$.  In practice, it is a good idea to set a maximum number of iterations in case the convergence is rather slow and the work cost is too high.
