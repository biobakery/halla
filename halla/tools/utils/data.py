import numpy as np
import math
from scipy.stats import rankdata, entropy
import pandas as pd

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
        raise ValueError('The argument should be a pandas DataFrame!')
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
            if item not in flag:
                flag[item] = counter
                counter += 1
            new_ar.append(flag[item])
        return(np.array(new_ar))

    def _discretize_continuous(ar, func=None, num_bins=None):
        '''Assign continuous vector to bins;
        currently only implement equal-frequency binning.
        # TODO: add more functions
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
        if func == 'equal-freq':
            order = rankdata(ar, method='min') - 1 # order starts with 0
            bin_size = np.ceil(len(ar) / float(num_bins))
            # rankdata to have increment of 1
            discretized_result = rankdata((order / bin_size).astype(int), method='dense') - 1
        # assign missing values to a separate bin
        discretized_result[np.isnan(discretized_result)] = np.nanmax(discretized_result) + 1
        return(discretized_result)
        
    # TODO: store available discretization functions somewhere
    if func not in [None, 'equal-freq']:
        raise ValueError('Discretization function not available...')
    if ar_type == object:
        return(_discretize_categorical(ar))
    return(_discretize_continuous(ar, func, num_bins))

def keep_feature(x, max_freq_thresh=0.8):
    '''Decide if the feature should be kept based on the maximum frequency threshold, given:
    - x              : the vector from a DataFrame row
    - max_freq_thresh: the threshold for maximum frequency in fraction [0..1]
    '''
    x = x.to_list()
    _, count = np.unique(x, return_counts=True)
    return(count.max() / count.sum() < max_freq_thresh)

def preprocess(df, types, max_freq_thresh=0.8, discretize_func=None, discretize_num_bins=None):
    '''Preprocess input data
    1) remove features with low entropy
    2) discretize values if needed

    Args:
    - df                 : a panda dataframe
    - types              : a numpy list indicating the type of each feature
    - max_freq_thresh    : the threshold for maximum frequency in fraction [0..1]
    - discretize_func    : function for discretizing
    - discretize_num_bins: # bins for discretizing #TODO: different bins for different features?
    '''
    # 1) remove features with low entropy
    kept_rows = df.apply(keep_feature, 1, max_freq_thresh=max_freq_thresh)
    types, df = types[kept_rows], df[kept_rows]

    # 2) discretize values if needed
    updated_df = df.copy(deep=True)
    for row_i, type_i in enumerate(types):
        updated_df.iloc[row_i] = discretize_vector(updated_df.iloc[row_i].to_numpy(), type_i,
                                                   func=discretize_func,
                                                   num_bins=discretize_num_bins)

    return(updated_df, df, types)

def is_all_cont(types):
    '''Check if all types are continuous
    '''
    return(np.sum(types == object) == 0)