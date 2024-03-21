import numpy as np


def entropy(array: np.ndarray) -> np.ndarray:
    """ Compute entropy values for each row of the given array. 
        
    Parameters
    ----------
    array : np.ndarray
        A 2d array of (signals, samples) containing integers between 0 and 
        n_unique_vals. It can also be a 1d array however, which implies a 
        single signal with all values corresponding to samples for that 
        signal. As well, if the array is not an integer dtype, the unique 
        values will first be computed in order to compute the entropy. 

    Returns
    -------
    np.ndarray
        An array of entropy values, one per signal.

    ..ref: https://stackoverflow.com/questions/33607071/fastest-way-to-compute-entropy-of-each-numpy-array-row

    """
    array = np.atleast_2d(array)
    if (array.dtype not in [np.int32, np.int64]) or (array.min() < 0):
        array = np.array([np.unique(a, return_inverse=True)[1] for a in array])
    
    nrows, ncols = array.shape
    nbins = array.max() + 1

    # count the number of occurrences for each unique integer 
    # between 0 and array.max() in each row of the array
    counts = np.vstack(list(np.bincount(a, minlength=nbins) for a in array))

    # divide by number of columns to get the probability of each unique value
    p = counts / float(ncols)

    # compute Shannon entropy in bits
    return -np.sum(p * np.log2(p), axis=1)
