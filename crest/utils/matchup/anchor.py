from collections.abc import Collection
from itertools import product
from .bruteforce import Grid
import numpy as np 




def anchor(grids: Collection[Grid]) -> (np.ndarray, np.ndarray):
    # Create trees and query against the reference (anchor) grid
    grids = product(grids[1:], grids[:1])
    match = list(map(get_matches, grids))
    n_ref = len(match[0])

    # Include the reference set indices in the final list of neighbors
    ix    = np.empty(n_ref, dtype=object)
    ix[:] = list(np.arange(n_ref, dtype=itype)[:, None])
    table = np.c_[[ix] + match]
    count = np.array([[1]*n_ref] + [list(map(len, m)) for m in match], dtype=itype)

    # Filter neighbor lists in which any of the grids are missing
    if (len(table) > 1) and (not allow_empty):
        empty = np.any(count[1:] == 0, 0)
        count = count[:, ~empty]
        table = table[:, ~empty]
    return table, count
