# HAllA revised version 0.0.1

Example codes can be found under `examples` directory.

## Loading data

### Handling missing data

- For continuous data, omit missing data in the similarity/distance computation
- For categorical data, assign missing values as a separate category

## Hierarchical clustering

Available pairwise distance metrics:
- `nmi`
- `pearson`
- `spearman`
- `dcor`

## Notes on installation

1. [Installing `rpy2` on macOS](https://stackoverflow.com/questions/52361732/installing-rpy2-on-macos)
2. Installing the necessary R packages

TODO: put these in the installation README

```
from rpy2.robjects.packages import importr

utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('EnvStats')
utils.install_packages('https://cran.r-project.org/src/contrib/Archive/eva/eva_0.2.5.tar.gz')
```
