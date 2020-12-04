import numpy as np
import math
from scipy.stats import rankdata, entropy, zscore, rankdata
from sklearn.preprocessing import KBinsDiscretizer, quantile_transform
import pandas as pd
import jenkspy
import warnings

def eval_type(df):
    '''Evaluate and set the type for each feature given dataframe df
       where each row represents one feature
    Return a tuple (updated_df, all_cont):
    - updated_df: df with updated type
    - types     : type for each row in np array
    '''
    def _get_type(np_array):
        '''Assess if the numpy array type is continuous
        #TODO: allow users to manually specify types, esp categorical integers
        '''
        try:
            np_array.astype(float)
            return(float)
        except:
            return(object)

    if not isinstance(df, pd.DataFrame):
        raise ValueError('The argument for eval_type() should be a pandas DataFrame!')
    # update all NaNs to None
    updated_df = df.copy(deep=True)
    types = []
    for row_i in range(updated_df.shape[0]):
        row_type = _get_type(updated_df.iloc[row_i].to_numpy())
        updated_df.iloc[row_i] = updated_df.iloc[row_i].to_numpy().astype(row_type)
        types.append(row_type)
    return(updated_df, np.array(types))

def discretize_vector(ar, ar_type=float, func=None, num_bins=None):
    '''Discretize vector to [0 .. (bin_num - 1)], given:
    - ar      : 1D numpy vector
    - ar_type : the type of the vector
    - func    : discretization function
    - num_bins: # bins
    '''
    def _discretize_categorical(ar):
        '''Represent the categories in integer
        '''
        new_ar = []
        flag, counter = {}, 0
        for item in ar:
            if item != item: item = 'nan'
            if item not in flag:
                flag[item] = counter
                counter += 1
            new_ar.append(flag[item])
        return(np.array(new_ar))

    def _discretize_continuous(ar, func=None, num_bins=None):
        '''Assign continuous vector to bins;
        available functions:
        - quantile, uniform, kmeans (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
        '''
        if func is None: # no discretization is needed
            return(ar)
        if num_bins is None:
            # by default, num_bins = sqrt(n) or # unique(n)
            num_bins = min(round(math.sqrt(len(ar))), len(set(ar)))
        elif num_bins == 0:
            raise ValueError('# bins should be > 0')
        else:
            num_bins = min(num_bins, len(set(ar)))
        if func in ['quantile', 'uniform', 'kmeans']:
            warnings.simplefilter('ignore') # ignore nan warning    
            # remove NaNs temporarily since KBinsDiscretizer currently doesn't handle NaNs
            temp_ar = np.array([x for x in ar if x == x])
            nans = np.array([not x == x for x in ar])
            discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy=func)
            temp_discretized_result = discretizer.fit_transform(temp_ar.reshape(len(temp_ar), 1)).reshape(len(temp_ar))
            # assign NaNs to a separate bin
            discretized_result = np.zeros(len(ar))
            discretized_result[~nans] = temp_discretized_result
            discretized_result[nans] = np.nan
            warnings.resetwarnings()
        elif func == 'jenks':
            warnings.simplefilter('ignore') # ignore nan warning    
            # remove the first lower bound
            breaks = jenkspy.jenks_breaks(ar, nb_class=num_bins)
            breaks[-1] += 1
            discretized_result = []
            for val in ar:
                if np.isnan(val):
                    discretized_result.append(np.nan)
                    continue
                for bound_i in range(1, len(breaks)):
                    if val >= breaks[bound_i-1] and val < breaks[bound_i]:
                        discretized_result.append(bound_i-1)
                        break
            discretized_result = np.array(discretized_result, dtype=float)
            warnings.resetwarnings()
        else:
            raise ValueError('Discretization function not available...')
        # tidy up data in case some bins are empty
        counter = 0
        for i in range(int(np.nanmax(discretized_result) + 1)):
            if i not in discretized_result: continue
            discretized_result[discretized_result == i] = counter
            counter += 1
        # assign missing values to a separate bin
        discretized_result[np.isnan(discretized_result)] = np.nanmax(discretized_result) + 1
        return(discretized_result)
        
    if ar_type == object:
        return(_discretize_categorical(ar))
    return(_discretize_continuous(ar, func, num_bins))

def keep_feature(x, max_freq_thresh=1):
    '''Decide if the feature should be kept based on the maximum frequency threshold, given:
    - x              : the vector from a DataFrame row
    - max_freq_thresh: the threshold for maximum frequency in fraction [0..1], disabled if None
    '''
    x = x.to_list()
    _, count = np.unique(x, return_counts=True)
    if max_freq_thresh is None: return(True)
    return(count.max() / count.sum() < max_freq_thresh)

def transform(df, types, funcs=None):
    '''Transform the continuous rows in the dataframe given function
    Available functions:
    - zscore, rank, quantile
    - any function available in numpy attributes, e.g., log, arcsin, sqrt
    '''
    warnings.filterwarnings('error') # to catch RuntimeError
    
    if funcs is None: return(df)
    
    if not isinstance(funcs, list): funcs = [funcs]
    funcs = [func.lower() for func in funcs]
    # ensure all functions are valid
    for func in funcs:
        if func not in ['zscore', 'rank', 'quantile'] and not hasattr(np, func):
            raise ValueError('The transform function should be an attribute of numpy or one of {zscore, rank, quantile}.')
    
    updated_df = df.copy(deep=True)
    for row_i, type_i in enumerate(types):
        if type_i == object: continue
        row = updated_df.iloc[row_i].to_numpy(dtype=float)
        for func in funcs:
            try:
                if func == 'zscore':
                    row = zscore(row, nan_policy='omit')
                elif func == 'rank':
                    row = rankdata(row)
                elif func == 'quantile':
                    row = quantile_transform(row.reshape((len(row), 1)), n_quantiles=len(row), output_distribution='normal', random_state=0).reshape(len(row))
                elif func == 'sqrt':
                    row = np.sqrt(np.abs(row)) * np.sign(row)
                else:
                    row = getattr(np, func)(row)
            except RuntimeWarning:
                raise ValueError('Runtime error in transforming data: invalid value encountered')
        updated_df.iloc[row_i] = row
    warnings.resetwarnings()
    return(updated_df)

def preprocess(df, types, transform_funcs=None, max_freq_thresh=0.8, discretize_func=None, discretize_num_bins=None):
    '''Preprocess input data
    1) remove features with low entropy
    2) transform if the function is specified
    3) discretize values if needed

    Args:
    - df                 : a panda dataframe
    - types              : a numpy list indicating the type of each feature
    - transform_funcs    : list of ordered functions to transform continuous data
    - max_freq_thresh    : the threshold for maximum frequency in fraction [0..1], disabled if None
    - discretize_func    : function for discretizing
    - discretize_num_bins: # bins for discretizing #TODO: different bins for different features?
    '''
    # 1) remove features with low entropy
    kept_rows = df.apply(keep_feature, 1, max_freq_thresh=max_freq_thresh)
    types, df = types[kept_rows], df[kept_rows]

    # 2) transform if the function is specified
    df = transform(df, types, funcs=transform_funcs)

    # 3) discretize values if needed
    updated_df = df.copy(deep=True)
    for row_i, type_i in enumerate(types):
        updated_df.iloc[row_i] = discretize_vector(updated_df.iloc[row_i].to_numpy().astype(type_i), type_i,
                                                   func=discretize_func,
                                                   num_bins=discretize_num_bins)

    return(updated_df, df, types)

def is_all_cont(types):
    '''Check if all types are continuous
    '''
    return(np.sum(types == object) == 0)