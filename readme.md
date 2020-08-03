# HAllA revised version

Note: the python setup is not yet finalized.

An example code `run_example.py` is currently provided under `halla` directory.

Some main changes compared to the original HAllA code include:

- the use of yaml for config, which is automatically converted into a `Struct` object by `config-loader.py`
- refactor HAllA into a class to enable both creating the class object and calling it via a terminal command

## Loading data

### Handling missing data

- For continuous data, omit missing data in the similarity/distance computation
- For categorical data, assign missing values as a separate category

## Hierarchical clustering

Available pairwise distance metrics:
- `nmi`
- `pearson`
- `spearman`

## Notes on installation

1. [Installing `rpy2` on macOS](https://stackoverflow.com/questions/52361732/installing-rpy2-on-macos)
2. Installing the necessary R packages

TODO: put these in the installation file

```
from rpy2.robjects.packages import importr

utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('EnvStats')
utils.install_packages('https://cran.r-project.org/src/contrib/Archive/eva/eva_0.2.5.tar.gz')
```