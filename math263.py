import numpy as np
import scipy as sp


def euler(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a) = y0
    over the interval [a, b] via n steps of Euler's method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    return x, y


def mem(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a)=y0
    over the interval [a, b] via n steps of Heun's modified Euler method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i + 1], y[i] + h * k1)
        y[i + 1] = y[i] + h * (k1 + k2) / 2
    return x, y


def bem(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a)=y0
    over the interval [a, b] via n steps of the backward Euler method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    for i in range(n):
        func = lambda Y: Y - (y[i] + h * f(x[i + 1], Y))
        y[i + 1] = sp.optimize.fsolve(func, y[i])
    return x, y


def rk4(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a)=y0
    over the interval [a, b] via n steps of the 4th order (classical) Runge–Kutta method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(x[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2 * (k2 + k3) + k4) / 6
    return x, y


def ab2(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a)=y0
    over the interval [a, b] via n steps of second order Adams–Bashforth method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    # take first step with Heun's MEM
    k1 = f(x[0], y[0])
    k2 = f(x[1], y[0] + h * k1)
    y[1] = y[0] + h * (k1 + k2) / 2
    # begin multistepping
    f2 = f(x[0], y[0])
    for i in range(1, n):
        f1 = f(x[i], y[i])
        y[i + 1] = y[i] + h * (3 * f1 - f2) / 2
        f2 = f1
        # step f-vals down to get ready for next step
    return x, y


def abm2(f, a, b, y0, n):
    """
    numerically solves the IVP
        y' = f(x,y), y(a)=y0
    over the interval [a, b] via n steps of second order Adams–Bashforth–Moulton
    predictor-corrector method
    """
    h = (b - a) / n
    x = np.linspace(a, b, num=n + 1)
    y = np.empty((x.size, np.size(y0)))
    y[0] = y0
    # starter method: Heun's modified Euler method
    k1 = f(x[0], y[0])
    k2 = f(x[1], y[0] + h * k1)
    y[1] = y[0] + h * (k1 + k2) / 2
    # continuing method: ABM2 predictor-corrector
    f2 = f(x[0], y[0])
    for i in range(1, n):
        # predict with AB2
        f1 = f(x[i], y[i])
        yhat = y[i] + h * (3 * f1 - f2) / 2
        # correct with AM1
        f0 = f(x[i + 1], yhat)
        y[i + 1] = y[i] + h * (f0 + f1) / 2
        # shift down f-val
        f2 = f1
    return x, y


def secant_method(func, x0, x1, maxiter, restol):
    y0 = func(x0)
    y1 = func(x1)
    if abs(y0) < restol:
        return x0
    if abs(y1) < restol:
        return x1
    for i in range(2, maxiter):
        x2 = x1 - y1 * (x1 - x0) / (y1 - y0)
        y2 = func(x2)
        if abs(y2) < restol:
            return x2
        x0, x1 = x1, x2
    return x2
