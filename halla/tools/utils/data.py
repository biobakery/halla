import numpy as np
import math
from scipy.stats import rankdata

def eval_type(df):
    '''Evaluate and set the type for each feature given dataframe df
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

    updated_df = df.copy(deep=True)
    types = []
    for row_i in range(updated_df.shape[0]):
        row_type = _get_type(updated_df.iloc[row_i].to_numpy())
        updated_df.iloc[row_i] = updated_df.iloc[row_i].to_numpy().astype(row_type)
        types.append(row_type)
    return(updated_df, np.array(types))

def discretize_vector(ar, ar_type=float, func=None, num_bins=None):
    '''Discretize vector, given:
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
            # TODO: handle missing data?
            order = rankdata(ar, method='min') - 1 # order starts with 0
            bin_size = np.ceil(len(ar) / float(num_bins))
            # rankdata to have increment of 1
            discretized_result = rankdata((order / bin_size).astype(int), method='dense')
        return(discretized_result)
        
    # TODO: store available discretization functions somewhere
    if func not in [None, 'equal-freq']:
        raise ValueError('Discretization function not available...')
    if ar_type == object:
        return(_discretize_categorical(ar))
    return(_discretize_continuous(ar, func, num_bins))
def preprocess(df, types, discretize_func=None, discretize_num_bins=None):
    '''Preprocess input data
    1) handle missing values # TODO
    2) discretize values if needed
    3) remove features with low entropy # TODO

    Args:
    - df                 : a panda dataframe
    - discretize_func    : function for discretizing
    - discretize_num_bins: # bins for discretizing #TODO: different bins for different features?
    '''
    updated_df = df.copy(deep=True)
    for row_i, type_i in enumerate(types):
        updated_df.iloc[row_i] = discretize_vector(updated_df.iloc[row_i].to_numpy(), type_i,
                                                   func=discretize_func,
                                                   num_bins=discretize_num_bins)

    return(updated_df)

def is_all_cont(types):
    '''Check if all types are continuous
    '''
    return(np.sum(types == object) == 0)