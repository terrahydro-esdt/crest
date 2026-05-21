from __future__ import annotations
from collections.abc import Collection, Iterator
from itertools import product
from numpy import unravel_index, ndarray


def partial_product(
    values : Collection[Collection], 
    start  : int | slice = 0, 
    stop   : int | None  = None,
    step   : int | None  = None,
) -> Iterator[tuple]:
    """ Efficiently index and slice a cartesian product.

    This method allows indexing / slicing a cartesian product without
    needing to compute intermediate values. For example, a cartesian
    product which generates one million samples could be indexed to 
    get the last sample without needing to compute all previous ones.

    Parameters
    ----------
    values : Collection[Collection]
        A collection of collections (e.g. list of numpy arrays) which 
        contains the values to create a cartesian product over.
    start  : int | slice
        Index to start iteration at (default is first value). Alternatively,
        a slice object which defines the indices of the product to output.
    stop   : int | None
        Index to stop interation at (default is all remaining values). Note
        that the value at this index is excluded, as with a slice object.
    step   : int | None
        Steps to take on each iteration (same as step parameter for slice).
        Note that intermediate step values are still calculated and discarded.

    Returns
    -------
    Iterator[tuple]
        Returns an iterator with tuples as values, where each tuple contains
        exactly one value from each of the input collections. Equivalent to 
        the output of itertools.product, albeit only outputs in [start, stop).

    Examples
    --------
    >>> from itertools import product
    >>> a = [[1,2,3], [4,5]]
    >>> p = list(product(*a))
    >>> p == list(partial_product(a))
    True
    >>> p[3:] == list(partial_product(a, 3))
    True
    >>> p[3:4] == list(partial_product(a, 3, 4))
    True
    >>> p[:2] == list(partial_product(a, stop=2))
    True
    >>> p[1::2] == list(partial_product(a, slice(1, None, 2)))
    True

    """
    def generator(vals: list[Collection], start: int) -> Iterator[tuple]:
        """ Stitches together multiple products to form the partial product.
        
        For example:
            partial_product([[1,2,3], [4,5,6]], 4) = 
                yield (2, 5)
                yield from product([2], [6])
                yield from product([3], [4,5,6])
        """

        # First yield the current product index we're on 
        idxs = unravel_index(start, list(map(len, vals)))
        yield tuple([v[i] for i, v in zip(idxs, vals)])

        # Then yield remaining partial products, iterating backwards
        for partial in range(len(idxs)-1, -1, -1):
            yield from product(*[
                [v[idx]]  if i <  partial else # Single values
                v[idx+1:] if i == partial else # Partial collection
                v                              # Complete collections
            for i, (idx, v) in enumerate(zip(idxs, vals))])

    if min(1, 1, *map(len, values)) < 1:
        return

    if isinstance(start, slice):
        step  = step or start.step
        stop  = stop or start.stop
        start = start.start

    if isinstance(start, ndarray):
        values = list(values)
        for i in start:
            yield from partial_product(values, i, i+1)
        return

    start = start or 0
    step  = step  or None

    for i, value in enumerate( generator(list(values), start) ):
        if (stop is not None) and (i >= (stop-start)): break
        if (step is not None) and ((i % step) != 0):   continue
        yield value
