import numba as nb
import numpy as np

TYPES = [nb.float32, nb.float64, nb.int32, nb.int64]


def lexsort(array: np.ndarray, inplace: bool = True) -> (np.ndarray, np.ndarray):
    """ Lexicographic sort of an array. 

    This is a faster version of np.lexsort, in some cases > 5x faster than 
    the native implementation. The general idea, aside from optimizing via
    numba, is that we can recursively order successive columns only when 
    duplicate values are encountered in previous columns. In this way, a
    significant amount of element sorting can be skipped over in general,
    as well as early stopping once a column has no duplicates. 
    
    Note: by default, the given array is modified inplace to avoid 
    unnecessary data copying. 

    Parameters  
    ----------
    array   : np.ndarray
        2d array to be lexicographically sorted.
    inplace : bool 
        Whether to modify the given array inplace (default), or make a 
        copy of it otherwise. 

    Returns
    -------
    (np.ndarray, np.ndarray)
        A tuple of two numpy arrays: the lexicographically sorted values,
        and the indices vector which specifies the new row order of the values.
        Note that the values array will have its rows lexicographically sorted
        with respect to its column values. 

    Examples
    --------
    >>> import numpy as np
    >>> a = np.random.randint(0, 5, size=21).reshape((7,3))
    >>> b, i = lexsort(a, inplace=False)
    >>> (i == np.lexsort(a.T[::-1])).all()
    True
    >>> (b == a[np.lexsort(a.T[::-1])]).all()
    True

    """
    if not inplace: array = array.copy()
    inds = np.arange(len(array), dtype=np.int32)
    lexsort_nb(array, inds, 0)
    return array, inds


@nb.njit([nb.boolean(fx[:]) for fx in TYPES], cache=False, inline='always', fastmath=True, nogil=True)
def is_sorted(a):
    """ Fast check whether array is sorted least to greatest """
    for i in nb.prange(len(a) - 1):
        if a[i] > a[i + 1]: return False
    return True


@nb.njit([nb.boolean(fx[:]) for fx in TYPES], cache=False, inline='always', fastmath=True, nogil=True)
def is_unique_sorted(a):
    """ Fast check whether array is sorted least to greatest, with only unique values """
    for i in nb.prange(len(a) - 1):
        if a[i] >= a[i + 1]: return False
    return True


""" 
Numba seems to have difficulty handling recursive functions in a parallel context. In order
for it to find the lexsort_nb definition inside the prange loop, a JIT function without 
parallelism must be defined first, and then the second parallelized version may be created.

The following two functions are exactly equivalent, except with parallel and cache set to False
for the first definition, and True for the second. It's likely the lexsort_nb call in the parallel
version is then calling the first (non-parallel) version, though this arrangement appears to be 
the limit of efficiency from brief testing (i.e. further manual unrolling of the parallel loop or
function definitions only slows down the overall runtime). 

Unfortunately, with cache=True, the first function needs to be compiled every time crest is 
run - which can add 10-20 seconds to the time to first batch. See here for more details:
    https://github.com/numba/numba/issues/6061#issuecomment-1216381263

"""


@nb.njit([(fx[:, :], nb.int32[:], nb.int32) for fx in TYPES], parallel=False, cache=False, fastmath=True, nogil=True)
def lexsort_nb(vals, inds, c=0):
    """ 
    Numba seems to have difficulty handling recursive functions in a parallel context. In order
    for it to find the lexsort_nb definition inside the prange loop, a JIT function without 
    parallelism must be defined first, and then the second parallelized version may be created.
    
    The following two functions are exactly equivalent, except with parallel and cache set to False
    for the first definition, and True for the second. It's likely the lexsort_nb call in the parallel
    version is then calling the first (non-parallel) version, though this arrangement appears to be 
    the limit of efficiency from brief testing (i.e. further manual unrolling of the parallel loop or
    function definitions only slows down the overall runtime). 
    
    Unfortunately, with cache=True, the first function needs to be compiled every time crest is 
    run - which can add 10-20 seconds to the time to first batch. See here for more details:
    https://github.com/numba/numba/issues/6061#issuecomment-1216381263
    
    """
    if (len(vals) > 1) and not is_unique_sorted(vals[:, c]):
        if not is_sorted(vals[:, c]):
            argsort = np.argsort(vals[:, c], kind='mergesort')
            vals[:] = vals[argsort]
            inds[:] = inds[argsort]

        if c < (vals.shape[1] - 1):
            groups = np.where(vals[:-1, c] != vals[1:, c])[0] + 1
            start = np.append([0], groups)
            end = np.append(groups, [len(vals)])
            for i in nb.prange(len(start)):
                lexsort_nb(vals[start[i]:end[i]], inds[start[i]:end[i]], c + 1)


@nb.njit([(fx[:, :], nb.int32[:], nb.int32) for fx in TYPES], parallel=True, cache=True, fastmath=True, nogil=True)
def lexsort_nb(vals, inds, c=0):
    """ 
    Numba seems to have difficulty handling recursive functions in a parallel context. In order
    for it to find the lexsort_nb definition inside the prange loop, a JIT function without 
    parallelism must be defined first, and then the second parallelized version may be created.
    
    The following two functions are exactly equivalent, except with parallel and cache set to False
    for the first definition, and True for the second. It's likely the lexsort_nb call in the parallel
    version is then calling the first (non-parallel) version, though this arrangement appears to be 
    the limit of efficiency from brief testing (i.e. further manual unrolling of the parallel loop or
    function definitions only slows down the overall runtime). 
    
    Unfortunately, with cache=True, the first function needs to be compiled every time crest is 
    run - which can add 10-20 seconds to the time to first batch. See here for more details:
    https://github.com/numba/numba/issues/6061#issuecomment-1216381263
    
    """
    if (len(vals) > 1) and not is_unique_sorted(vals[:, c]):
        if not is_sorted(vals[:, c]):
            argsort = np.argsort(vals[:, c], kind='mergesort')
            vals[:] = vals[argsort]
            inds[:] = inds[argsort]

        if c < (vals.shape[1] - 1):
            groups = np.where(vals[:-1, c] != vals[1:, c])[0] + 1
            start = np.append([0], groups)
            end = np.append(groups, [len(vals)])
            for i in nb.prange(len(start)):
                lexsort_nb(vals[start[i]:end[i]], inds[start[i]:end[i]], c + 1)
