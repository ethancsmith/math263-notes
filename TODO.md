# TODO

1. Consider moving systems and higher order ODEs up before multistep methods.
1. Store f-vals in an array for continuing methods to avoid shifting data. 
1. Rewrite methods to initialize y-arrays differently depending on whether y0 is scalar or vector.
1. Simplify pyplot code.
1. Switch to using $t$ as generic independent variable?
1. Include RKF45 example of adaptive methods.
1. Revisit IPython display of Markdown output.
	- Works fine in Jupyter Lab.
	- Jupyter Book does not handle this correctly at present. See [Open issue](https://github.com/executablebooks/jupyter-book/issues/1771).
	- Try using {eval} tags to embed computed results in Markdown cells? Requires the following in the top-matter of the notebook.
		``` yaml
			mystnb:
 	   			execution_mode: 'inline'
		```

## Reference collection

1. Adaptive methods
	1. [RKF45](https://ntrs.nasa.gov/api/citations/19690021375/downloads/19690021375.pdf)
	1. [DP54 (aka, RK45)](https://www.sciencedirect.com/science/article/pii/0771050X80900133)
