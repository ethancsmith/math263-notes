# MATH 263 Notes

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<YOUR URL HERE>)

Code and lecture notes for MATH 263 (numerical differential equations) at GCC

## Build instructions

Set up your environment with `conda` and the included environment file.

```shell
conda env create -f environment.yml
conda activate math263
```

The MyST Markdown scripts can now be used with in Jupyter Lab as Jupyter notebooks.
Simply Launch Jupyter Lab.
Then right-click on the MD file and select "Open With" to open the file as a "Notebook."
Alternatively, you can build the IPYNB version first using `make`.

```shell
make notebooks
```

To build everything (including the HTML version of the Jupyter Book) simply `make`.

```shell
make
```
