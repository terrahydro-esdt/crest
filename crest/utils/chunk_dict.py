from __future__ import annotations
from itertools import starmap, zip_longest
from typing import Iterator, Iterable
from math import ceil
from tlz import merge, partition_all


def chunk_dict(n: int, d: dict | Iterable[dict]) -> Iterator[dict]:
    """Chunk a dict of lists into separate dicts with lists of max length `n`.
    
    Parameters
    ---------
    n : int
        Chunk size, i.e. max length of the new dictionary values.
    d : dict
        Dictionary of iterables to chunk.

    Returns
    -------
    Iterator[dict]
        Dictionaries whose values are now of length at most `n`.

    Examples
    --------
    >>> import tlz
    >>> c = [{'a':1, 'b':2}, {'a':3, 'b':4}, {'a':5, 'b':6}]
    >>> d = tlz.merge_with(list, c)
    >>> print(d)
    {'a': [1, 3, 5], 'b': [2, 4, 6]}
    >>> e = chunk_dict(2, d) # Now chunk to a certain size
    >>> print(list(e))
    [{'a':[1,3], 'b':[2,4]}, {'a':[5], 'b':[6]}]

    >>> import numpy as np
    >>> size3 = [{'a': np.array([1,2,3]), 'b': np.array([4,5,6])}, 
    ...          {'a': np.array([7]),     'b': np.array([8])}]
    >>> merge = tlz.merge_with(np.hstack, size3) # Merge batches together
    >>> print(merge)
    {'a': array([1, 2, 3, 7]), 'b': array([4, 5, 6, 8])}
    >>> size2 = chunk_dict(2, merge) # Invert merge_with using a different size
    >>> for batch in size2:
    ...     print(batch)
    {'a': array([1, 2]), 'b': array([4, 5])}
    {'a': array([3, 7]), 'b': array([6, 8])}
    
    """

    def chunk(k, v):
        """ Allows slicing numpy arrays so they remain array objects """
        if hasattr(v, '__len__') and hasattr(v, '__getitem__'):
            try:    return ({k: v[i*n:(i+1)*n]} for i in range(ceil(len(v)/n)))
            except: pass # objects may still not support slicing
        return ({k: part} for part in partition_all(n, v))
    return map(merge, zip_longest(*starmap(chunk, d.items()), fillvalue={}))