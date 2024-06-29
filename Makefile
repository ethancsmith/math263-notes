SRCNAMES = 01-direction_fields 02-euler_method 03-error_analysis 04-runge-kutta_methods 05-multistep_methods
SCRIPTS = $(addsuffix .md, $(SRCNAMES))
NOTEBOOKS = $(addsuffix .ipynb, $(SRCNAMES))

all: $(NOTEBOOKS) book

%.ipynb: %.md
	jupytext --sync $<

book: $(NOTEBOOKS) _config.yml _toc.yml 00-intro.md
	jb build .

clean:
	git clean -xf

cleanall:
	git clean -xfd

.PHONY: all book clean cleanall
