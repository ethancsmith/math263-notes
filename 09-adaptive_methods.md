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

# Lecture 9: Adaptive methods.

## Adaptive methods.
Up to this point, all of our methods have operated with a fixed step-size $h$. **Adaptive methods** incorporate a procedure to adjust the step-size in an attempt to keep the truncation error under control.  When we use numerical methods in practice, however, we typically cannot solve for the true solution exactly, and so it is impossible to compute the error.  Adaptive methods, therefore, incorporate some procedure to estimate the local truncation error at each step.  

## Estimating the unknown.

Here we give a description of how adaptive methods estimate local truncation error – something that is typically unknowable.  As we will see, the idea is to use a second numerical method of one higher order of accuracy.  Suppose that we have 2 methods for numerically integrating an ODE.  For simplicity, we assume that both are single step methods.  We assume that the first method takes the form
\begin{equation*}
y_{i+1} = y_i + h\phi_1(x_i, y_i, h),
\end{equation*}
while the second takes the form
\begin{equation*}
\hat{y_{i+1}} = \hat{y_i} + h\phi_2(x_i,\hat{y_i}, h).
\end{equation*}
Furthermore, we assume that the first method has local truncation error $l_{i+1}=O(h^{p+1})$  while the second is $\hat{l_{i+1}}=O(h^{p+2})$, i.e., if $y$ is the true solution, then we have
\begin{align*}
y(x_{i+1}) &= y(x_i) + h\phi_1(x_i, y(x_i), h) + O(h^{p+1}),\\
y(x_{i+1}) &= y(x_i) + h\phi_2(x_i, y(x_i), h) + O(h^{p+2}).
\end{align*}
Since we are interested in the local truncation error, we assume that previous step was exact, i.e., we assume $y_i = y(x_i) = \hat{y_i}$ exactly.  Then the local truncation error for the first method is
\begin{equation*}
l_{i+1} = y(x_{i+1}) - y_i - h\phi_1(x_i, y_i, h) 
= y(x_{i+1}) - y_{i+1}
\end{equation*}
while the local truncation error for the second method is
\begin{equation*}
\hat{l_{i+1}} = y(x_{i+1}) - \hat{y_i} - h\phi_1(x_i, \hat{y_i}, h) 
= y(x_{i+1}) - \hat{y_{i+1}},
\end{equation*}
and so
\begin{equation*}
l_{i+1} = \hat{l_{i+1}} + (\hat{y_{i+1}} - y_{i+1}).
\end{equation*}
However, since $l_{i+1} = O(h^{p+1})$ and $\hat{l_{i+1}} = O(h^{p+2})$, it is reasonable to assume that the local truncation error for the first (lower order) method
\begin{equation*}
l_{i+1}\approx \hat{y_{i+1}} - y_{i+1}
\end{equation*}
when $h$ is small. 

## Controlling error.

Treating the truncation error as a function of $h$, we assume that $l_{i+1}(h) \approx K h^{p+1}$ for some constant $K$.  Suppose then that we are willing to adjust the step size from $h$ to $\delta h$ so as to keep the average local truncation error under some tolerance $\epsilon$, i.e., we want to choose $\delta$ so that
\begin{equation*}
\left|\frac{l_{i+1}(\delta h)}{\delta h}\right|\le \epsilon.
\end{equation*}
Observing that our assumptions imply that 
\begin{equation*}
l_{i+1}(\delta h) \approx K(\delta h)^{p+1} \approx \delta^{p+1} l_{i+1}(h) \approx \delta^{p+1}|\hat{y_{i+1}} - y_{i+1}|
\end{equation*}
and solving for $\delta$, we have
\begin{equation*}
\delta\le \left(\frac{\epsilon h}{|\hat{y_{i+1}} - y_{i+1}|}\right)^{1/p}.
\end{equation*}
In practice, $\delta$ is usually chosen somewhat more conservatively since there is a "work" penalty to be paid every time that the step has to be repeated with a smaller step-size.

## Advantages and disadvantages.

An adaptive method offers the advantage of being able to estimate the local truncation error at each step of the algorithm and adjust the step size $h_i$ to control the estimated error.  The major disadvantages are that adaptive methods are typically more computationally expensive, and they are typically much harder to implement than non-adaptive methods. 

## Runge–Kutta–Fehlberg.

One of the most popular adaptive methods employing this scheme is the **Runge-Kutta-Fehlberg method (RKF45)**, which uses an order 4 Runge-Kutta method (but not the classical one) together with an order 5 Runge-Kutta method to estimate and control the error.  The two methods were chosen so as to share as many of the evaluations of the right-hand side $f(x,y)$ as possible.  In general, an order 4 method requires 4 functional evaluations and an order 5 method requires another 6 for a total of 10.  The RKF45 method, however, requires only 6 total.

## Example.

TODO.

```{code-cell}
#
```
