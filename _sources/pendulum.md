---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: math263-notes
  language: python
  name: python3
---

# Physical example: Simple gravity pendulum
$\renewcommand{\dee}{\mathrm{d}}$

+++

Under certain simplifying assumptions, the motion of a swinging pendulum is described by the ODE
```{math}
:label: pendulum-ode
\frac{\dee^2\theta}{\dee t^2} + \frac{g}{L}\sin\theta = 0,
```
where $L$ is the length of the pendulum, $g$ is the acceleration due to gravity near the Earth's surface, and $\theta$ is the angle that the pendulum makes with the vertical axis.

Below we study this model numerically over a 10-second interval when $g=9.8$ m/s<sup>2</sup> and $L=5$ m.
Our first step is to convert the second-order ODE {eq}`pendulum-ode` to a first-order vector ODE.
To that end, let $\boldsymbol u = \langle \theta, \dot\theta\rangle$ and $\boldsymbol f(t, \boldsymbol u) = \langle \dot\theta, -\frac{g}{L}\sin\theta\rangle$.
Then {eq}`pendulum-ode` is equivalent to the first-order (vector) ODE
\begin{equation*}
\frac{\dee\boldsymbol u}{\dee t} = \boldsymbol f(t, \boldsymbol u).
\end{equation*}

```{code-cell} ipython3
# import modules
import numpy
from IPython.display import HTML
from matplotlib import animation, pyplot
from numpy import pi

import math263
```

```{code-cell} ipython3
# define IVP params
g = 9.8  # acceleration due gravity near surface of Earth (m/s^2)
L = 5  # length of pendulum rod (m)


# rewrite higher-order ODE as first-order system u' = f(t, u), where u = <theta, theta'>
# u' = <theta', theta''> = <theta', -(g/l)sin(theta)> = f(t, u)
def f(t, u):
    theta, Dtheta = u
    return numpy.array([Dtheta, -(g / L) * numpy.sin(theta)])


# study over a 10-second time-interval 
a, b = 0, 10
# set initial angle and angular velocity
theta0 = pi / 4
Dtheta0 = 0
u0 = numpy.array([theta0, Dtheta0])

# numerically solve with time-steps of length h = 0.1 secs
h = 0.1
n = int((b - a) / h)
t, u = math263.rk4(f, a, b, u0, n)

# extract angles and angular velocities at each time t_i
theta = u[:, 0]
Dtheta = u[:, 1]
```

```{code-cell} ipython3
# make various plots of numerical solution
pyplot.style.use("dark_background")
fig = pyplot.figure()
fig.set_size_inches(10, 7.5)
title = f"""
$\\ddot{{\\theta}} + \\frac{{g}}{{L}}\\sin\\theta = 0,\\quad$ \
$g = {g}\\mathregular{{\\ m/s^2}},\\quad$ \
$L = {L}$ m
$\\theta({a}) = {theta0:0.4f},\\quad$ \
$\\dot{{\\theta}}({a}) = {Dtheta0:0.4f}\\mathregular{{\\ 1/s}}$
"""
fig.suptitle(title)

# plot theta (angle) vs t (time)
ax = fig.add_subplot(2, 1, 1)
color = "blue"
ax.plot(t, theta, color=color, label=r"$\theta$")
ax.set_xlabel(r"$t$ (sec)")
ax.set_ylabel(r"$\theta$", color=color)
ax.grid(True)
ax.tick_params(axis="y", labelcolor=color)

# plot Dtheta (angular velocity) vs t (time) with
ax = ax.twinx()
color = "green"
ax.plot(t, Dtheta, color=color, label=r"$\dot\theta$")
ax.set_ylabel(r"$\dot\theta$ (1/sec)", color=color)
ax.grid(True)
ax.tick_params(axis="y", labelcolor=color)

# make phase portrait theta' vs. theta
ax = fig.add_subplot(2, 1, 2)
ax.plot(theta, Dtheta, color="orange")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\dot{\theta}$ (1/sec)")
ax.grid(True)
# slide second plot to right to avoid covering up vertical axis label
# box = ax.get_position()
# box.x0 += 0.01
# ax.set_position(box)

pyplot.show()
```

```{code-cell} ipython3
# compute physical coordinates of the "bob" at each time t_i
x = L * numpy.sin(theta)
y = -L * numpy.cos(theta)

# compute edges of window in xy-plane
xmin, xmax = min(x), max(x)
ymin, ymax = min(y), 0
xmin -= 0.1
xmax += 0.1
ymin -= 0.5

#pyplot.style.use("default")
#pyplot.xkcd()
#pyplot.rcParams["font.family"] = ["Comic Sans MS"]
fig, ax = pyplot.subplots()
ax.set_title("Pendulum")

rod = ax.plot([0, x[0]], [0, y[0]], color="blue")[0]  # draw "rod"
bob = ax.scatter(x[0], y[0], color="red", marker="o")  # draw "bob"
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
ax.set_xlabel("$x$ (meters)")
ax.set_ylabel("$y$ (meters)")
pyplot.gca().set_aspect('equal')


def frame_update(frame):
    rod.set_xdata([0, x[frame]])
    rod.set_ydata([0, y[frame]])
    bob.set_offsets([x[frame], y[frame]])
    return rod, bob


# we make a new frame for each time-node
# we pause for h * 1000 milliseconds before drawing the next frame
anim = animation.FuncAnimation(
    fig=fig, func=frame_update, frames=len(t), interval=h * 1000
)
pyplot.close()
HTML(anim.to_html5_video())
```

```{code-cell} ipython3

```
