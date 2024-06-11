import numpy as np

def euler(f, a, b, y0, n):
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

def mem(f, a, b, y0, n):
	'''
	numerically solves the IVP
		y' = f(x,y), y(a)=y0
	over the interval [a, b] via n steps of Heun's modified Euler method 
	'''
	h = (b-a)/n;
	x = np.linspace(a, b, num=n+1);
	y = np.empty(x.size);
	y[0] = y0;
	for i in range(n):
		k1 = f(x[i], y[i]);
		k2 = f(x[i+1], y[i]+h*k1);
		y[i+1] = y[i]+h*(k1+k2)/2;

	return (x, y)

def ab2(f, a, b, y0, n):
	'''
	numerically solves the IVP
		y' = f(x,y), y(a)=y0
	over the interval [a, b] via n steps of second order Adams–Bashforth method
	'''
	h = (b-a)/n;
	x = np.linspace(a, b, num=n+1);
	y = np.empty(x.size);
	y[0] = y0;
	# take first step with Heun's MEM
	k1 = f(x[0], y[0]);
	k2 = f(x[1], y[0]+h*k1);
	y[1] = y[0]+h*(k1+k2)/2;
	# begin multistepping
	f2 = f(x[0], y[0]);
	for i in range(1, n):
		f1 = f(x[i], y[i]);
		y[i+1] = y[i]+h*(3*f1-f2)/2;
		f2 = f1; # step f-vals down to get ready for next step

	return (x, y)

def rk4(f, a, b, y0, n):
	'''
	numerically solves the IVP
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
