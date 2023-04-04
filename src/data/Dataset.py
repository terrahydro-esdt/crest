from dask.diagnostics import ProgressBar
from collections.abc import Iterable
from fsspec.mapping import FSMap 
from contextlib import nullcontext
from functools import partial
from pathlib import Path 

import numpy as np 
import dask.array as da
import dask

from crest.src.base import BaseSet
from crest.src.data import Datafile, Blockset


class Dataset(BaseSet):
    """Wraps a collection of Datafiles, thus loading multiple sources. 
    
    Parameters
    ----------
    locations : Iterable[Datafile | Path | str | FSMap]
        The set of Datafiles or data locations which form this 
        Dataset. If any paths are passed in via this parameter,
        a new Dataset object is created to load from that path.
        Because this inherits from BaseSet, functions existing 
        in the Datafile class can be called by this object in 
        order to apply the function across all Datafiles in this 
        set. See crest.src.base.BaseSet for more details.
    **kwargs
        Any additional keyword arguments are passed into each
        newly created Datafile, thus allowing global options
        to be set across Datafiles when creating the Dataset.
        Note that this is only used for locations which are not
        already passed in as a Datafile object.
    
    """
    def __init__(self, locations: Iterable[Datafile|Path|str|FSMap], **kwargs):
        self.container = list(map(partial(Datafile.load, **kwargs), locations))


    def generate_samples(self, 
        numblocks : list[int] | None = None,
        verbose   : bool = True,
    ) -> da.Array:
        """Generate the dask array containing all valid samples.

        Notes
        -----
        Sequence of steps to prepare the data:
            1. Ensure all Datafiles are aware of all dimensions
            2. Deptermine number of blocks per dimension and element overlaps
            3. Rechunk data based on the required number of blocks
            4. Apply overlap, create Blocksets with one Block per Datafile

        Sequence of steps to find valid samples (in parallel across Blocksets):
            1. Find valid windows for each Block (based on Block.valid_percent)
            2. Match windows between Blocks that are within a certain distance
            3. Create a lazy dask array containing indices of matched windows
            4. Combine arrays from all Blocksets into the final dask array

        Parameters
        ----------
        numblocks : list[int] | None 
            A list of integers, one per dimension, which defines the number of
            blocks each dimension should be split into. Blocks subset the data
            and are processed in parallel, leading to faster runtime and a 
            smaller memory footprint; too many blocks can cause slowdown due to
            overhead, however. Number of blocks must also result in the same
            number of elements per block, minus the last block; e.g. [1,2,3,4]
            cannot be divided into 3 blocks because that would lead to 
            [[1,2],[3],[4]]. By default, the number of blocks is determined 
            automatically based on data chunking - but this can be suboptimal.
        verbose   : bool
            Whether or not logs should be shown when generating samples.

        Returns
        -------
        dask.Array
            Lazy array which contains all valid windows which were found. 

        """
        if verbose: print('Preparing data...')

        # Ensure all Datafiles are aware of all dimensions
        self.ensure_dims( set.union(*map(set, self.dims)) )

        # With one value per dimension:
        max_res = np.nanmax(self.resolution, axis=0) 
        idx_res = np.nanargmax(self.resolution, axis=0)
        tgt_blk = numblocks or np.gcd.reduce(self.numblocks, axis=0)
        skipdim = idx_res == np.arange(len(self))[:, None]

        if verbose:
            print('\tcurrent blocks:', self.numblocks)
            print('\tmax resolution:', max_res)
            print('\t target blocks:', tgt_blk)

        # Calculate required overlaps for each Datafile
        overlap = self.calculate_overlap(max_res, _map=[skipdim])
        if verbose: print('\tblock overlaps:', overlap)

        # Update Datafile chunks to create the required number of blocks
        self.update_chunks(tgt_blk)
        if verbose: print('\t result blocks:', self.numblocks[0])

        # Apply the required overlaps to each Datafile
        prepped = self.apply_overlap(_map=[overlap])
        lengths = list(map(len, prepped))
        assert(len(set(lengths)) <= 1), lengths
        if verbose: print(f'\tSplit into {lengths[0]} total block(s)')

        # Create a list of Blocksets, where each Blockset 
        # contains exactly one Block from each Datafile
        blocksets = list(map(Blockset, zip(*prepped)))

        # Find all samples in parallel across the created blocksets
        with nullcontext() if not verbose else ProgressBar():
            if verbose: print('\nFinding all valid samples...')
            delayed = lambda blockset: dask.delayed(blockset.find_matches)()
            results = da.compute( *map(delayed, blocksets) )
            samples = da.hstack(results)
        
        if verbose: print(f'\nFound {len(samples):,} samples')
        return samples 
