import numpy as np

def discretize(matrix):
    return(matrix)

def eval_type(df):
    '''Evaluate and set the type for each feature given dataframe df
    Return a tuple (updated_df, all_cont):
    - updated_df: df with updated type
    - types     : type for each row in np array
    '''

    def get_type(np_array):
        '''Assess if the numpy array type is continuous
        #TODO: is integer continuous / categorical?
        '''
        try:
            np_array.astype(float)
            return(float)
        except:
            return(object)

    updated_df = df.copy(deep=True)
    types = []
    for row_i in range(updated_df.shape[0]):
        row_type = get_type(updated_df.iloc[row_i].to_numpy())
        updated_df.iloc[row_i] = updated_df.iloc[row_i].to_numpy().astype(row_type)
        types.append(row_type)
    return(updated_df, np.array(types))

def preprocess(matrix):
    '''Preprocess input data #TODO
    1) handle missing values
    2) discretize values if needed
    3) remove features with low entropy

    Arg:
    - matrix: a panda dataframe
    '''
    
    return(matrix)

def is_all_cont(types):
    '''Check if all types are continuous
    '''
    return(np.sum(types == object) == 0)