SRCNAMES = 01-direction_fields 02-euler_method 03-error_analysis 04-runge-kutta_methods 05-multistep_methods
SCRIPT_EXTN = md # jupyter notebooks are generated from MyST Markdown scripts
SCRIPTS = $(addsuffix .$(SCRIPT_EXTN), $(SRCNAMES))
NOTEBOOKS = $(addsuffix .ipynb, $(SRCNAMES))

all: $(NOTEBOOKS) book

%.ipynb: %.$(SCRIPT_EXTN)
	jupytext --sync $<

book: $(NOTEBOOKS) _config.yml _toc.yml 00-intro.md
	jb build .

notebooks: $(NOTEBOOKS)

clean:
	git clean -xf

cleanall:
	git clean -xfd

.PHONY: all book clean cleanall notebooks
