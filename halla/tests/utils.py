import numpy as np

def compare_numpy_array(ar1, ar2):
    '''Compare and check if the values of two numpy arrays are equal
    '''
    all_float = False
    try:
        ar1 = ar1.astype(float)
        ar2 = ar2.astype(float)
        all_float = True
        np.testing.assert_allclose(ar1, ar2, verbose=True)
        return(True)
    except:
        if all_float: return(False)
        try:
            np.testing.assert_array_equal(ar1, ar2)
            return(True)
        except:
            return(False)