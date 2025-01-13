SRCNAMES = 01-direction_fields 02-euler_method 03-error_analysis 04-runge-kutta_methods 05-multistep_methods 06-predictor-corrector_methods 07-systems_of_odes 08-higher_order_odes 09-adaptive_methods 10-shooting_method 11-finite_differences 12-stiffness
SCRIPT_EXTN = md # jupyter notebooks are generated from MyST Markdown scripts
SCRIPTS = $(addsuffix .$(SCRIPT_EXTN), $(SRCNAMES))
NOTEBOOKS = $(addsuffix .ipynb, $(SRCNAMES))

all: $(NOTEBOOKS) book

%.ipynb: %.$(SCRIPT_EXTN) math263.py
	jupytext --sync $<

book: $(NOTEBOOKS) _config.yml _toc.yml 00-intro.md
	jb build .

notebooks: $(NOTEBOOKS)

deploy: # deploy web version to gh-pages branches
	ghp-import -n -p -f _build/html

clean:
	git clean -xf

cleanall:
	git clean -xfd

.PHONY: all book clean cleanall deploy notebooks
