import numpy as np

def feuler(f, a, b, y0, n):
	'''
	numerically solves the IVP
		y' = f(x,y), y(a)=y0
	over the interval [a, b] via n steps of the (foward) Euler method 
	'''
	h = (b-a)/n;
	x = np.linspace(a, b, num=n+1);
	y = np.empty(x.size);
	y[0] = y0;
	for i in range(n):
		y[i+1] = y[i]+h*f(x[i], y[i]);

	return (x, y)

def rk4(f, a, b, y0, n):
	'''
	numerical solve the IVP
		y' = f(x,y), y(a)=y0
	over the interval [a, b] via n steps of the 4th order (classical) Runge–Kutta method 
	'''
	h = (b-a)/n
	x = np.linspace(a, b, num=n+1);
	y = np.empty(x.size);
	y[0] = y0;
	for i in range(n):
		k1 = f(x[i], y[i])
		k2 = f(x[i]+h/2, y[i]+h*k1/2)
		k3 = f(x[i]+h/2, y[i]+h*k2/2)
		k4 = f(x[i]+h, y[i]+h*k3)
		y[i+1] = y[i]+h*(k1+2*(k2+k3)+k4)/6;
	return (x, y)
