# TODO

- [ ] Consider moving systems and higher order ODEs up before multistep methods.
- [ ] Add discusion and example of creating direction fields for systems and second-order ODEs in the "phase plane."
- [ ]  Store f-vals in an array for continuing methods to avoid shifting data. Not sure this is a win.
- [ ] Rewrite methods to initialize y-arrays differently depending on whether y0 is scalar or vector.
- [ ] Clean and simplify pyplot code.
	- [ ] 1: Direction fields
	- [ ] 2: Euler's method
	- [ ] 3: Error analysis
	- [ ] 4: Runge—Kutta methods
	- [ ] 5: Multistep methods
	- [ ] 6: Predictor-corrector
	- [x] 7: Systems of ODEs
	- [ ] 8: Higher-order ODEs
	- [ ] 9: Adaptive methods
	- [ ] 10: Shooting method
	- [ ] 11: Finite differences
	- [ ] 12: Stiffness
- [x] Switch to using $t$ as generic independent variable?
- [ ] Include RKF45 example of adaptive methods.
- [ ] Revisit IPython display of Markdown output.
	- Works fine in Jupyter Lab.
	- Jupyter Book does not handle this correctly at present. See [Open issue](https://github.com/executablebooks/jupyter-book/issues/1771).
	- Try using {eval} tags to embed computed results in Markdown cells? Requires the following in the top-matter of the notebook.
		``` yaml
			mystnb:
 	   			execution_mode: 'inline'
		```
	- [Glue](https://jupyterbook.org/en/stable/content/executable/output-insert.html) is another option to consider. Could keep the Python glue code in a separate cell and ["remove-input"](https://myst-nb.readthedocs.io/en/latest/render/hiding.html).
1. Save a lecture: Fold analysis of Euler's method into lecture on Euler's method, and just develop order 2 RK methods at once (i.e., MEM isn't separate).
1. Add a lecture on well-posedness or on Taylor series methods before RK methods?

## Reference collection

1. Adaptive methods
	1. [RKF45](https://ntrs.nasa.gov/api/citations/19690021375/downloads/19690021375.pdf)
	1. [DP54 (aka, RK45)](https://www.sciencedirect.com/science/article/pii/0771050X80900133)
