import numpy as np

def feuler(f, a, b, y0, n):
	'''
	numerically solves IVP
		y' = f(x,y), y(a)=y0
	over the interval [a, b] via n steps of the (foward) euler method 
	'''
	h = (b-a)/n;
	x = np.linspace(a, b, num=n+1);
	y = np.empty(x.size);
	y[0] = y0;
	for i in range(n):
		y[i+1] = y[i] + h * f(x[i], y[i]);

	return (x, y)
