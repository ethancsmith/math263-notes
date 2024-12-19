
# Lecture 9: Adaptive methods.

## Adaptive methods.
Up to this point, all of our methods have operated with a fixed-step size $h$. **Adaptive methods** incorporate a procedure to adjust the step size in an attempt to keep the truncation error under control.  When we use numerical methods in practice, however, we typically cannot solve for the true solution exactly, and so it is impossible to compute the error.  Adaptive methods, therefore, incorporate some procedure to estimate the local truncation error at each step.  

## Estimating the unknown.

Here we give a description of how adaptive methods estimate local truncation error -- something that is typically unknowable.  As we will see, the idea is to use a second numerical method of one higher order of accuracy.  Suppose that we have 2 methods for numerically integrating an ODE.  For simplicity, we assume that both are single step methods.  We assume that the first method takes the form
```{math}
y_{i+1} = y_i + h\phi_1(x_i, y_i, h),
```
while the second takes the form
```{math}
\hat{y_{i+1}} = \hat{y_i} + h\phi_2(x_i,\hat{y_i}, h).
```


## Notes to self

We will not implement an adaptive method.  Instead we will use the SciPy library's scipy.integrate.solve_ivp.  I think that RK45 is what I have been calling RKF45.

```{code-cell} ipython3

```
