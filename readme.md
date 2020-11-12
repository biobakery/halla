# HAllA revised version 0.0.7

Example codes can be found under `examples` directory. More details can be found on the [wiki](https://github.com/biobakery/halla_revised/wiki) page.

## Installation

1. Other than [Python](https://www.python.org/) (version >= 3.7) and [R](https://www.r-project.org/) (version >= 3.6.1), install all required libraries listed in `requirements.txt`, specifically:

- [jenkspy](https://github.com/mthh/jenkspy) (version >= 0.1.5)
- [Matplotlib](https://matplotlib.org/) (version >= 3.3.0)
- [NumPy](https://numpy.org/) (version >= 1.19.0)
- [pandas](https://pandas.pydata.org/) (version >= 1.0.5)
- [PyYAML](https://pypi.org/project/PyYAML/) (version >= 5.3.1)
- [rpy2](https://pypi.org/project/rpy2/) (version >= 3.3.5) - [Notes on installing `rpy2` in macOS](https://stackoverflow.com/questions/52361732/installing-rpy2-on-macos)
- [scikit-learn](https://scikit-learn.org/stable/) (version >= 0.23.1)
- [SciPy](https://www.scipy.org/) (version >= 1.5.1)
- [seaborn](https://seaborn.pydata.org/) (version >= 0.10.1)
- [statsmodels](https://www.statsmodels.org/stable/index.html) (version >= 0.11.1)
- [tqdm](https://github.com/tqdm/tqdm) (version >= 4.50.2)

Users can either install them one-by-one or install all of them at once by running:

```
# for MacOS - read the notes on installing rpy2:
#   specifically run 'env CC=/usr/local/Cellar/gcc/X.x.x/bin/gcc-X pip install rpy2'
#   where X.x.x is the gcc version on the machine **BEFORE** running the following command
pip install -r requirements.txt
```

2. Install with `setup.py`

```
python setup.py install
```

## Available parameters

Available pairwise distance metrics:
- `nmi`
- `pearson`
- `spearman`
- `dcor`
- `xicor`
