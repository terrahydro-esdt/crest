from collections.abc import Collection, Callable
from functools import cached_property
from numbers import Number

import dask.dataframe as dd
import dask.array as da 
import numpy as np 
import logging
import time 

from crest.base import BaseSet
from crest.utils import find_neighbors, Stopwatch
from .SampleSet import SampleSet
from .Block import Block


class Blockset(BaseSet):
    """Wraps a collection of Blocks into a single object.

    Parameters
    ----------
    blocks : Collection[Block]
        The set of Blocks which form this Blockset. Because
        this inherits from BaseSet, functions existing in the
        Block class can be called by this object in order to
        apply the function across all Blocks in this set. See
        crest.crest.base.BaseSet for more details.

    """
    def __init__(self, 
        blocks  : Collection[Block] | Collection[Callable], 
        logger  : logging.Logger | None = None,
        timing  : bool = True,
        shuffle : bool = False,
    ):
        self.container = [getattr(b, '__call__', lambda: b)() for b in blocks]
        self.logger  = logger or logging.getLogger('Blockset')
        self.timing  = timing and (logger is not None)
        self.shuffle = shuffle
        
        # Set block count for all blocks
        for i, block in enumerate(self.container):
            block.block_count = len(self.container)
            block.block_index = i
            if self.timing:
                block.logger = self.logger


    @cached_property
    def dtype(self):
        """ dtype for the dask sample array, and for each Sample element """
        return np.dtype([(f'Data_{i}', b.dtype) for i,b in enumerate(self)])


    @cached_property
    def benchmark(self):
        """ Return a Stopwatch function for benchmarking """
        return lambda label, log=self.logger.debug: Stopwatch(
            prefix=f'\t\tBlockset.{label}',
            logger=log,
            silent=not self.timing,
        )


    def find_matches(self, task_bytes: Number = 1e9) -> da.Array:
        """ Build the BallTree and find all valid samples """
        # Create meta/dtype information for dask
        meta = np.empty(0, dtype=self.dtype)

        # Benchmark timing for data loading / neighbor finding
        with self.benchmark('find_matches') as timer:
            timer.prefix += ' | 100% of time spent loading data'

            # Fast return when there are no valid windows for a block
            with self.benchmark('fast_invalid_check'):
                for block in self:
                    if block.fast_invalid_check:
                        return da.from_array(meta)

            # Return when there are no valid windows for a block
            with self.benchmark('valid_windows.size'):
                for block in self:
                    if not block.valid_windows.size:
                        return da.from_array(meta)

            loading_time = time.time() - timer.start['time']

            # Sort blocks by the number of valid windows, so that the Block with
            # the fewest windows is used as the BallTree query reference. This
            # is a proxy for the overall coarsest resolution Block, as coarsest
            # resolution could be found in different Blocks when there are multiple
            # dimensions (e.g. Block_1 has coarsest dim_1, and Block_2 with dim_2)
            # self.sort(lambda block: max(block.resolution))
            # self.sort(lambda block: -block.valid_windows.size)
            
            # Find neighboring points between coordinate grids for the valid window
            # locations, within the resolution tolerances provided
            with self.benchmark('find_neighbors'):
                matches, counts = find_neighbors(
                    self.valid_coords, 
                    self.valid_resolution,
                    logger=self.logger if self.timing else None,
                    shuffle=self.shuffle,
                )

            complete_time = time.time() - timer.start['time']
            loading_pct   = (loading_time / complete_time) * 100
            timer.prefix  = timer.prefix.replace('100', f'{loading_pct:.0f}')

        # Return if there aren't any matches
        if counts.size < 1: return da.from_array(meta)

        with self.benchmark('_group_matches'):
            matches, divisions, lengths = self._group_matches(matches, counts, task_bytes)
        
        # Create a dask dataframe first, then transform into a dask
        # array, in order to satisfy dask's built in assumptions 
        return dd.from_map(self._parse, matches, **{
            'meta'             : (0, int), 
            'token'            : f'product{id(matches)}',
            'divisions'        : [0] + divisions.tolist(), 
            'enforce_metadata' : False,
            'singleton'        : len(lengths) == 1,
        }).to_dask_array(lengths=list(lengths), meta=meta)


    def _parse(self, matches, singleton: bool = False) -> SampleSet:
        """ Parse a match into the relevant SampleSet of data """
        with self.benchmark('_parse.extract'):
            windows = self.extract(_map=[matches]) # Extract data windows
            # ordered = self.sort(container=windows) # Return to original ordering
        return SampleSet(list(zip(*windows)), singleton, self.dtype)
        

    def _group_matches(self, matches: np.ndarray, counts: np.ndarray, task_bytes: Number):
        """ Combine matches into larger groups for higher throughput """
        # Number of samples in the cartesian product for dataframe divisions
        cartesian = np.prod(counts, axis=0)
        divisions = np.cumsum(cartesian, dtype=counts.dtype)

        # Combine multiple match sets together into a single SampleSet for 
        #   faster processing, with up to `task_bytes` of data per SampleSet
        # This also drastically reduces memory and time used by dd.from_map
        maxim = task_bytes / self.dtype.itemsize # Can have up to 2x numerator
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
        return matches, divisions, cartesian