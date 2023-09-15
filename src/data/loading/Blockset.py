from collections.abc import Collection, Callable
from functools import cached_property

import dask.dataframe as dd
import dask.array as da 
import numpy as np 

from crest.src.base  import BaseSet
from crest.src.utils import find_neighbors, Stopwatch
from crest.src.data.loading import Block, SampleSet


class Blockset(BaseSet):
    """Wraps a collection of Blocks into a single object.

    Parameters
    ----------
    blocks : Collection[Block]
        The set of Blocks which form this Blockset. Because
        this inherits from BaseSet, functions existing in the
        Block class can be called by this object in order to
        apply the function across all Blocks in this set. See
        crest.src.base.BaseSet for more details.

    """
    def __init__(self, blocks: Collection[Block] | Collection[Callable], logger=print):
        self.container = [getattr(b, '__call__', lambda: b)() for b in blocks]
        self.logger = logger 

        # Set block count for all blocks
        for block in self.container:
            block.block_count = len(self.container)


    @cached_property
    def dtype(self):
        """ dtype for the dask sample array, and for each Sample element """
        return np.dtype([(f'Data_{i}', b.dtype) for i,b in enumerate(self)])


    def find_matches(self) -> da.Array:
        """ Build the BallTree and find all valid samples """
        # Create meta/dtype information for dask
        meta  = np.empty(0, dtype=self.dtype)

        # Fast return when there are no valid locations for this block
        if any(self.fast_invalid_check):     return da.from_array(meta)
        if not all(self.valid_windows.size): return da.from_array(meta)

        # Sort blocks by the number of valid windows, so that the Block with
        # the fewest windows is used as the BallTree query reference. This
        # is a proxy for the overall coarsest resolution Block, as coarsest
        # resolution could be found in different Blocks when there are multiple
        # dimensions (e.g. Block_1 has coarsest dim_1, and Block_2 with dim_2)
        # self.sort(lambda block: max(block.resolution))
        self.sort(lambda block: -block.valid_windows.size)
        # print([b.valid_windows.shape for b in self])
        
        # Find neighbors for the valid window locations within 1/2 the
        # resolution of the reference, using Chebyshev distance (L-inf)
        matches, counts = find_neighbors(self.valid_coords, [b.valid_resolution for b in self], p=np.inf)

        # Clean up memory resources that aren't needed beyond this point
        # Not currently used, since the objects in memory will be used
        # shortly after this while batching the samples
        self.cleanup()

        # Return if there aren't any matches
        if counts.size < 1: return da.from_array(meta)

        # Number of samples in the cartesian product for dataframe divisions
        cartesian = np.prod(counts, axis=0)
        divisions = np.cumsum(cartesian, dtype=counts.dtype)

        # Combine multiple match sets together into a single SampleSet for 
        #   faster processing, with up to 20MB of data per SampleSet
        # This also drastically reduces memory and time used by dd.from_map
        maxim = 1e7 / self.dtype.itemsize # Can have up to 2x numerator
        total = divisions[-1]
        first = divisions[0]
        n_ele = min(maxim, total)

        # Check that groups actually need combining to reach size threshold
        if n_ele > cartesian.min():

            # Determine unique indices of the boundaries where matches should split
            boundaries = np.arange(max(first+1, n_ele), total, n_ele)
            partitions = np.searchsorted(divisions, boundaries)
            unique_idx = np.flatnonzero(np.diff(partitions, prepend=-1))

            # Create combined match sets and recompute dataframe divisions
            matches   = np.split(matches, partitions[unique_idx], axis=1)
            hist_bins = np.r_[0, boundaries[unique_idx], total]
            cartesian = np.histogram(divisions, hist_bins, weights=cartesian)[0]
            divisions = np.cumsum(cartesian.copy(), dtype=counts.dtype)

        # Otherwise just reshape to mimic having 1 set of matches per SampleSet
        else: matches = matches.T[..., None]
        
        # Create a dask dataframe first, then transform into a dask
        # array, in order to satisfy dask's built in assumptions 
        return dd.from_map(self._parse, matches, **{
            'meta'             : (0, int), 
            'token'            : f'product{id(matches)}',
            'divisions'        : [0] + divisions.tolist(), 
            'enforce_metadata' : False,
            'singleton'        : len(cartesian) == 1,
        }).to_dask_array(lengths=list(cartesian), meta=meta)


    def _parse(self, matches, singleton: bool = False) -> SampleSet:
        """ Parse a match into the relevant SampleSet of data """
        windows = self.extract(_map=[matches]) # Extract data windows
        ordered = self.sort(container=windows) # Return to original ordering
        return SampleSet(list(zip(*ordered)), singleton, self.dtype)
        