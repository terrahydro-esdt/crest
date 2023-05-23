from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from collections.abc import Collection
from itertools import zip_longest, product, starmap
from numbers import Number, Integral as Int

import dask.dataframe as dd
import dask.array as da 
import dask 
import xarray as xr 
import pandas as pd
import numpy as np 

from crest.src.base  import BaseSet
from crest.src.utils import find_neighbors
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
    def __init__(self, blocks: Collection[Block]):
        self.container = blocks 


    def find_matches(self) -> da.Array:
        """ Build the BallTree and find all valid samples """

        # Create meta/dtype information for dask
        dtype = np.dtype([(f'Data_{i}', T) for i, T in enumerate(self.dtype)])
        meta  = np.empty((0,), dtype=dtype)

        # Fast return when there are no valid locations for this block
        if any(self.fast_invalid_check):     return da.from_array(meta)
        if not all(self.valid_windows.size): return da.from_array(meta)

        # Sort blocks by the number of valid windows, so that the Block with
        # the fewest windows is used as the BallTree query reference. This
        # is a proxy for the overall coarsest resolution Block, as coarsest
        # resolution could be found in different Blocks when there are multiple
        # dimensions (e.g. Block_1 has coarsest dim_1, and Block_2 with dim_2)
        self.sort(lambda block: -block.valid_windows.size)

        # Find neighbors for the valid window locations within 1/2 the
        # resolution of the reference, using Chebyshev distance (L-inf)
        matches = find_neighbors(self.valid_coords, self.resolution, p=np.inf)
        self.cleanup()

        # Return if there aren't any matches
        if not len(matches): return da.from_array(meta)

        # Count total number of matches
        counts = np.prod([list(map(len, m)) for m in matches], axis=1)

        # Create dataframe divisions
        divisions = [0] + list(np.cumsum(counts))

        # Create dask dataframe and convert to da.Array
        return dd.from_map(self._parse, matches, **{
            'meta'             : (0, int), 
            'token'            : f'product{id(matches)}',
            'divisions'        : divisions, 
            'enforce_metadata' : False,
            'singleton'        : len(counts) == 1,
        }).to_dask_array(lengths=list(counts), meta=meta)


    def _parse(self, match: Collection[np.ndarray], singleton: bool) -> SampleSet:
        """ Parse a match into the relevant SampleSet of data """
        windows = self.extract(_map=[match])   # Extract data windows
        ordered = self.sort(container=windows) # Return to original ordering
        return SampleSet(ordered, singleton)
        