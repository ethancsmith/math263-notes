import numpy as np

def feuler(f, a, b, y0, n):
	'''
	numerically solves IVP
		y' = f(x,y), y(0)=y0
	over the interval [a, b] via n steps of the (foward) euler method 
	'''
	h = (b-a)/n;
	x = np.linspace(a, b, num=n+1);
	y = np.zeros(x.size);
	y[0] = y0;
	for k in range(n):
		y[k+1] = y[k] + h * f(x[k], y[k]);

	return (x, y)
