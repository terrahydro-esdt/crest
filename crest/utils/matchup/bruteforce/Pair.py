from collections.abc import Collection
from contextlib import nullcontext
import numpy as np
import logging
import re 

# Allow bruteforce progress logging if numba_progress available
try:                from numba_progress import ProgressBar
except ImportError: ProgressBar = None 

from .Grid  import Grid
from .utils import bruteforce_double, StreamingLogger
from crest.utils.matchup.bruteforce.utils.entropy import entropy



class Pair:
    def __init__(self, G1: Grid, G2: Grid):
        self.G1 = G1
        self.G2 = G2


    def __repr__(self) -> str: 
        return f'Pair({self.G1}, {self.G2})'


    def match_indices(self, 
        logger   : logging.Logger | None = None,
        optimize : bool = True, 
        debug    : bool = False,
    ) -> np.ndarray:
        """ Find indices from G1 and G2 where rows of the two are matches """
        # Duplicate grids as necessary to align with each other
        G1 = self.G1.tiled_align(self.G2)
        G2 = self.G2.tiled_align(self.G1)
        assert(G1.ndim == G2.ndim), [G1.ndim, G2.ndim]
        assert(G1.dims == G2.dims), [G1.dims, G2.dims]

        # Find any all-NaN (virtual) columns in the grids
        virtual_1 = np.isnan(G1.coordinates).all(0)
        virtual_2 = np.isnan(G2.coordinates).all(0)
        skip_dims = virtual_1 | virtual_2

        # If optimizing, swap grids as necessary to use optimal order
        if optimize:

            # Check virtual dims count (i.e. all values in column are NaN):
            #  Grid with fewer samples is first when same number of virtuals
            #  Otherwise, grid with more virtual dims is first
            v1, v2 = virtual_1.sum(), virtual_2.sum()
            switch = (len(G1)>len(G2)) if v1==v2 else (v1<v2)
            if debug: print(f'\t{v1=} {v2=} {switch=}')
            if switch:
                self.G1, self.G2 = self.G2, self.G1
                G1, G2 = G2, G1

        # If optimizing, reorder the columns as necessary to use optimal order
        if optimize:

            # Order columns by the first grid column's entropy values
            cols = np.argsort([np.inf if skip else entropy(col)[0] for skip,col
                               in zip(skip_dims, G1.coordinates.T)])
            row_order_1, G1 = G1.reorder(cols)
            row_order_2, G2 = G2.reorder(cols)
            skip_dims = skip_dims[cols]

        if debug:
            print('  b q shape:', G1.coordinates.shape, G2.coordinates.shape)
            print('br qr shape:', G1.resolutions.shape, G2.resolutions.shape)
            print('  skip dims:', skip_dims)

        # Perform the actual matchup procedure
        progress = nullcontext(None)
        if debug or (logger is not None):
            if ProgressBar is not None:
                streamer = None if debug else StreamingLogger(logger, logging.DEBUG)
                progress = ProgressBar(
                    total=len(b), 
                    file=streamer, 
                    update_interval=10, 
                    dynamic_ncols=False,
                    notebook=False, 
                    postfix='find_neighbors progress',
                )

        with progress as pbar:
            ix1, ix2 = bruteforce_double(G1.coordinates, G2.coordinates, 
                                        *G1.resolutions, *G2.resolutions,
                                         skip_dims, pbar)

        # Recover the original row indices
        if optimize:
            ix1 = row_order_1[ix1]
            ix2 = row_order_2[ix2]
        return ix1, ix2


    def combined_grid(self, 
        ix1 : np.ndarray,
        ix2 : np.ndarray,
        deduplicate : bool = True,
    ) -> Grid:
        """ Construct a combination Grid of G1 and G2 where they match """
        
        def matched(ix: np.ndarray, G: Grid) -> Grid:
            """ Create a grid using the matching indices """
            def resolutions():
                # If either grid uses a non-uniform resolution, both need to
                R = G.resolutions
                if self.G1.anisotropic or self.G2.anisotropic:
                    if R.shape[1] == 1: 
                        R = np.tile(R, (1, len(G), 1))
                    return R[:, ix]
                return R
            return G.clone(**{
                'coordinates' : G.coordinates[ix], 
                'resolutions' : resolutions,
                'table'       : G.table[ix],
            })

        # Create new Grids using the matching indices 
        M1 = matched(ix1, self.G1)
        M2 = matched(ix2, self.G2)

        # To remove duplicates, we split into the sub-grids and check equality
        if deduplicate:
            M2s = M2.split()
            M1s = [m for m in M1.split() if m not in M2s]
                               
            # Table, index, and name should not change due to deduplication
            return Grid.combine(M1s + M2s, **{
                'table' : lambda: np.c_[M1.table, M2.table],
                'index' : M1.index + M2.index,
                'name'  : f'{M1.name}+{M2.name}',
            })
        return Grid.combine([M1, M2])



    @classmethod
    def select_pair(cls, grids: Collection[Grid], debug: bool=False) -> 'Pair':
        """ Choose two from the given Grid list and return the Pair object """
        # First check for grids that are from the same source
        pattern = re.compile(r'Block\[\d+\:(.*)_.+\]')
        sources = [(pattern.findall(G.name) + [None])[0] for G in grids]
        isvalid = lambda s: (s is not None) and (sources.count(s) > 1)
        options = list(filter(isvalid, set(sources)))

        if options:
            ix1 = sources.index(options[0])
            ix2 = sources.index(options[0], ix1+1)
            lbl = 'Sources'

        # Next check if there are any uniform resolution grids
        elif sum(uniform := [not G.anisotropic for G in grids]) > 1:
            ix1 = uniform.index(True)
            ix2 = uniform.index(True, ix1+1)
            lbl = 'Uniform'

        # Otherwise, just select the last two grids
        else: lbl, ix1, ix2 = 'Default', -1, -1
        
        if debug: print(f'[{lbl}] Selected indices {ix1} and {ix2}')
        return cls(grids.pop(ix2), grids.pop(ix1))