from collections.abc import Collection, Sequence
from functools import cached_property, partial
from typing import Union
from typing import Callable

import dask.dataframe as dd
import dask.array as da 
import numpy as np 
import logging
import numbers
import time 

from crest.base import BaseSet
from crest.utils import find_neighbors, Stopwatch
from .SampleSet import SampleSet
from .Block import Block


# Some type checkers report Literal numbers aren't compatible with Number
Number = numbers.Real | int | float


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
        blocks  : Union[Collection[Block] | Collection[Callable]],
        logger  : Union[logging.Logger, None] = None,
        timing  : bool = True,
        shuffle : bool = False,
    ):
        self.container = [getattr(b, '__call__', lambda: b)() for b in blocks]
        self.logger  = logger or logging.getLogger('Blockset')
        self.timing  = timing and (logger is not None)
        self.shuffle = shuffle
        self.dropped = [] 
        self.dropped_features = []
        self.dtype = np.dtype([(f'Data_{i}', b.dtype) for i,b in enumerate(self)])

        # Set block count for all blocks
        for i, block in enumerate(self.container):
            setattr(block, 'block_count', len(self.container))
            setattr(block, 'block_index', i)
            if self.timing: 
                setattr(block, 'logger', self.logger)


    def __repr__(self) -> str:
        return f'Blockset[{"".join(str(b.block_index) for b in self)}]'

    # @cached_property
    # def dtype(self):
    #     """ dtype for the dask sample array, and for each Sample element """
    #     return np.dtype([(f'Data_{b.block_index}', b.dtype) for b in self])


    @cached_property
    def benchmark(self):
        """ Return a Stopwatch function for benchmarking """
        return lambda label, logger=self.logger.debug, **kwargs: Stopwatch(**({
            'message' : f'\t\t{self}.{label}',
            'logger'  : logger,
            'silent'  : not self.timing,
        } | kwargs))


    def exclude(self, names: list[str], features):
        """ Exclude certain Datafile Blocks from this Blockset """
        blocks = Blockset([], self.logger, self.timing, self.shuffle)
        keep = [b for b in self if not any(name in b.label for name in names)]
        drop = [b for b in self if     any(name in b.label for name in names)]
        setattr(blocks, 'container', keep)
        setattr(blocks, 'dropped',   drop)
        setattr(blocks, 'dropped_features', [b.feature_subset(features) for b in drop])
        setattr(blocks, 'dtype', self.dtype)
        assert(len(blocks)), f'All Blocks excluded from {self} using {names}'
        return blocks


    def find_matches(self, 
        task_bytes      : Number = 1e9, 
        n_block_samples : int | None = None,
        verbose         : bool = False,
        valid_pct = None,
        features  = None,
        dropped   = None,
    ) -> da.Array:
        """ Build the BallTree and find all valid samples """
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            self.timing = True

        # Create meta/dtype information for dask
        meta = np.empty(0, dtype=self.dtype)

        if dropped is not None:
            self = self.exclude(dropped, features)

        # Set new valid percent configurations
        if valid_pct is not None:
            for name, vp in valid_pct.items():
                for block in self:
                    if name in block.label:
                        block.set_valid_percent(vp)

        # Benchmark timing for data loading / neighbor finding
        with self.benchmark('find_matches') as timer:
            timer.message += ' | 100% of time spent loading data'

            # Fast return when there are no valid windows for a block
            with self.benchmark('fast_invalid_check'):
                for block in self:
                    if block.fast_invalid_check:
                        self.logger.debug(f'\t{block} failed fast_invalid_check')
                        return da.from_array(meta)

            # Return when there are no valid windows for a block
            with self.benchmark('valid_windows.size'):
                for block in self:
                    if not block.valid_windows.size:
                        self.logger.debug(f'\t{block} failed valid_windows.size')
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
                    grid_labels = [str(b) for b in self],
                    axis_labels = [b.dims for b in self],
                    logger  = self.logger if self.timing else None,
                    shuffle = self.shuffle,
                    debug   = False,
                )

            complete_time = time.time() - timer.start['time']
            loading_pct   = (loading_time / complete_time) * 100
            timer.message = timer.message.replace('100', f'{loading_pct:.0f}')

        # Return if there aren't any matches
        if counts.size < 1: return da.from_array(meta)

        task_mb = f'Task={task_bytes/1e6:.0f} MB'
        with self.benchmark(f'_grouped (shape={matches.shape} | {task_mb})'):
            matches, divs, lengths = self._grouped(matches, counts, task_bytes, n_block_samples)

        # Undo the valid_percent modification(s)
        if valid_pct is not None:
            self.reset_valid_percent()

        # Create a dask dataframe first, then transform into a dask
        # array (in order to satisfy dask's built in assumptions)
        return dd.from_map(self._parse, matches, lengths, **{
            'meta'             : (0, int), 
            'token'            : f'product{id(matches)}',
            'divisions'        : [0] + divs.tolist(), 
            'enforce_metadata' : False,
            'features'         : self.feature_subset(features),
            'make_objs'        : features is None,
            'singleton'        : len(lengths) == 1,
        }).to_dask_array(lengths=list(lengths), meta=meta)


    def _parse(self, 
        matches   : Union[Sequence, np.ndarray],
        n_samples,# : int,
        features,#  : list[list[str]],
        make_objs : bool,
        singleton : bool,
    ) -> SampleSet:
        """ Parse a match into the relevant SampleSet of data.
        
        Notes
        -----
        If features are known ahead of time and are given, then the output
        representation is rigid and therefore the process can be much faster.
        Otherwise, Sample objects are created for each element in order to
        contain all information and enable a flexible representation.            

        """
        with self.benchmark(f'_parse.extract (shape={matches[0].shape})'):
            windows = self.extract(_map=[matches, features]) # Extract data windows
            # ordered = self.sort(container=windows) # Return to original ordering

        # Re-add dropped Block features as NaN
        for block, block_features in zip(self.dropped, self.dropped_features):
            block_window = block.extract(np.zeros(1), block_features, empty=True)
            sample_count = len(matches[0])

            if make_objs: 
                block_window = [block_window] * sample_count
            else:         
                block_window, dtype = block_window
                block_window = (np.array([block_window] * sample_count), dtype)
            windows.container.insert(block.block_index, block_window)

        # Zip the matches from each Block together to form the final Samples
        if make_objs: windows, dtypes = list(zip(*windows)), self.dtype
        else:         windows, dtypes = list(zip(*windows))
        return SampleSet(windows, int(n_samples), make_objs, singleton, dtypes)
        

    def _grouped(self, 
        matches         : np.ndarray, 
        counts          : np.ndarray, 
        task_bytes      : Number,
        n_block_samples : int | None,
    ):
        """ Combine matches into larger groups for higher throughput """
        # Number of samples in the cartesian product for dataframe divisions
        cartesian = np.prod(counts, axis=0)
        divisions = np.cumsum(cartesian, dtype=counts.dtype)

        # Combine multiple match sets together into a single SampleSet for 
        #   faster processing, with up to `task_bytes` of data per SampleSet
        # This also drastically reduces memory and time used by dd.from_map
        maxim = n_block_samples or (task_bytes / self.dtype.itemsize)
        total = divisions[-1]
        first = divisions[0]
        n_ele = int(min(maxim, total))

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
