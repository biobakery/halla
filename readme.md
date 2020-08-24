# HAllA revised version 0.0.6

Example codes can be found under `examples` directory. More details can be found on the [wiki](https://github.com/biobakery/halla_revised/wiki) page.

## Installation

1. Install all required libraries listed in `requirements.txt` - [Notes on installing `rpy2` in macOS](https://stackoverflow.com/questions/52361732/installing-rpy2-on-macos)

```
pip install -r requirements.txt
```

2. Install with `setup.py`

```
python setup.py install
```

3. Install the necessary R packages; run on Python

```
from rpy2.robjects.packages import importr

utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('EnvStats')
utils.install_packages('https://cran.r-project.org/src/contrib/Archive/eva/eva_0.2.5.tar.gz')
# check if eva has been successfully installed
eva = importr('eva')
```

## Available parameters

Available pairwise distance metrics:
- `nmi`
- `pearson`
- `spearman`
- `dcor`
