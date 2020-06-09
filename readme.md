# HAllA revised version

Note: the python setup is not yet finalized.

An example code `run_example.py` is currently provided under `halla` directory.

Some main changes compared to the original HAllA code include:

- the use of yaml for config, which is automatically converted into a `Struct` object by `config-loader.py`
- refactor HAllA into a class to enable both creating the class object and calling it via a terminal command

### Loading data

TODO:
- What to do with missing data?
- Test on real data to further 'generalize' parsing

### Hierarchical clustering

Available pairwise distance metrics:
- `nmi`
- `pearson`
- `spearman`
