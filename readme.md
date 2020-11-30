# ***ATTENTION***

Before opening a new issue here, please check the appropriate help channel on the bioBakery Support Forum (https://forum.biobakery.org) and consider opening or commenting on a thread there.

----
# HAllA revised version 0.0.7
Given two high-dimensional 'omics datasets X and Y (continuous and/or categorical features) from the same `n` biosamples, HAllA (Hierarchical All-against-All Association Testing) discovers **densely-associated blocks** of features in the X vs. Y association matrix where:

- each block is defined as all associations between features in a subtree of X hierarchy and features in a subtree of Y hierarchy,
- and a block is densely associated if (1 - FNR)% of pairwise associations are FDR significant (FNR is the pre-defined expected false negative rate)

<p align="center">
  <img src="https://user-images.githubusercontent.com/7066351/89912733-e3c89a80-dbc0-11ea-81c4-a696b321f150.png" width="750">
</p>

Example codes can be found under `examples` directory. 

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

## HAllA Overview

### Available parameters

Available pairwise distance metrics:
- `nmi`
- `pearson`
- `spearman`
- `dcor`

There are three steps in HAllA:

1. [Computing pairwise similarity matrix between all features in X and Y](#1-pairwise-similarity-matrix-computation)
2. [Hierarchical clustering of features in X and Y separately](#2-hierarchical-clustering)
3. [Finding densely-associated blocks iteratively](#3-finding-densely-associated-blocks)

### 1. Pairwise similarity matrix computation

The pairwise similarity matrix between all features in X and Y is computed with a specified similarity measure, such as Spearman correlation and normalized mutual information (NMI). This step then generates the p-value and q-value tables.

Note that for handling heterogeneous data, all continuous features are first **discretized** into bins using a specified binning method.

### 2. Hierarchical clustering

Hierarchical clustering on the features in each dataset is performed using the converted similarity measure used in step 1. It produces a tree for each dataset.

### 3. Finding densely-associated blocks

This recursive step is described in the pseudocode below:

```
def find_densely_associated_blocks(x, y):
    x_features = all features in x
    y_features = all features in y
    if is_densely_associated(x_features, y_features):
        report block and terminate
    else:
        # bifurcate one according to Gini impurities of the splits
        x_branches, y_branches = bifurcate_one_of(x, y)
        if both x and y are leaves:
            terminate
        for each x_branch and y_branch in x_branches and y_branches:
            find_densely_associated_blocks(x_branch, y_branch)

initial function call: find_densely_associated_blocks(X_root, Y_root)
```

For example, given two datasets of X (features: X1, X2, X3, X4, X5) and Y (features: Y1, Y2, Y3, Y4) both hierarchically clustered in X tree and Y tree, the algorithm first evaluates the roots of both trees and checks if the block consisting of all features of X and Y are densely-associated (if %significance (%reject) >= (1 - FNR)%).

<p align="center">
  <img src="https://user-images.githubusercontent.com/7066351/89906358-461d9d00-dbb9-11ea-946a-f729ac700ad4.png" width="500">
</p>

If the block is not densely-associated, the algorithm would bifurcate one of the trees. It would pick one of:

- [X1 X2][X3 X4 X5] >< [Y1 Y2 Y3 Y4] or
- [X1 X2 X3 X4 X5] >< [Y1 Y2 Y3][Y4]

based on the Gini impurity of the splits (pick the split that produces a lower weighted Gini impurity),

<p align="center">
  <img src="https://user-images.githubusercontent.com/7066351/89912316-6b61d980-dbc0-11ea-8373-779ab9e4aa35.png" width="500">
</p>

Once it picks the split with the lower impurity (let's say the first split), it will iteratively evaluate the branches:
- find densely-associated blocks in [X1 X2] vs [Y1 Y2 Y3 Y4], and
- find densely-associated blocks in [X3 X4 X5] vs [Y1 Y2 Y3 Y4]

and keep going until it terminates.
