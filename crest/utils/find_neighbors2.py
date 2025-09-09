from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from collections.abc import Collection, Callable
from contextlib import nullcontext
from itertools import combinations, product, chain, starmap
from functools import reduce, partial, cached_property

# Allow bruteforce progress logging if numba_progress available
try:                from numba_progress import ProgressBar
except ImportError: ProgressBar = None 

import numpy as np
import polars as pl
import pandas as pd 
import logging 

from ._bruteforce import *
from .print_table import print_table
from .Stopwatch import Stopwatch
from crest.utils.matchup.bruteforce.utils.entropy import entropy  
# from .lexsort import lexsort


def full_resolutions(coordinates: np.ndarray, resolutions: np.ndarray):
    """ Resolutions expanded into their full generic representation.

    The general format for resolutions is to have two resolution vectors
    for each coordinate vector: one for the left (lower) bound, and one
    for the right (upper) coordinate bound.
    
    There are three cases that must be handled:
    1. self.resolutions.ndim == 1
        This is a uniform resolution (i.e. all points use the same
        resolution vector), and so we can simply duplicate the left/right
        side and have a singleton dimension for the coordinate rows. For
        example, given data with 3 dimensions, we would have resolutions
        shape=(3,), which we tile and return a shape of (2, 1, 3).
    2. self.resolutions.ndim == 2
        This is a non-uniform resolution (i.e. all points use a different
        resolution vector), and so we only need to shift this left/right
        by one element to create the lower/upper bounds.
    3. self.resolutions.ndim == 3 
        This is an anisotropic resolution, where the left/right bounds
        for each coordinate row aren't necessarily the same coming from
        different directions. For example, given the following situation:
            - point A matches point X if (A-A_left <= X <= A+A_right)
            - point B matches point X if (B-B_left <= X <= B+B_right)
            - point B is the direct neighbor of point A to the right
        With a non-uniform resolution, A_right == B_left, since resolutions
        can change per point, but are only shifted left or right by one. In
        contrast, for the anisotropic case, we can have A_right != B_left,
        such that neighboring points can have overlapping regions in which
        they would match a point (or equivalently, regions between them for
        which neither point would match).
    
    Returns
    -------
    np.ndarray
        Full resolution representation which has three dimensions:
        (2, n_coordinates or 1, n_dimensions). The first dimension
        represents the left/right bounds; the second dimension is
        the resolution for each coordinate row (duplicated as 1 if
        using a uniform resolution); and the third dimension is the
        same as the number of coordinate dimensions. 

    """
    r_shape = resolutions.shape 
    n_coord = len(coordinates)
    n_dims  = coordinates.shape[-1]

    # Uniform resolution, simply duplicate for left/right side
    if resolutions.ndim == 1:
        assert(r_shape[0] == n_dims), (
            f'Expected resolution vector to contain {n_dims} ' +
            f'elements; found shape: {r_shape}')

        # Just tile for left/right and use 1 for the coordinate length
        full = np.tile(resolutions, (2, 1, 1))
    
    # Non-uniform resolution, extend to endpoints for left and right
    elif resolutions.ndim == 2:
        # Resolution should have one fewer elements, as these represent 
        # both the right and left resolution for neighboring coordinates
        # (i.e. resolutions sit between coordinate elements)
        assert(r_shape[1] == (n_coord-1)), (
            f'Expected resolutions shaped ({n_dims}, {n_coord-1}); ' +
            f'found shape: {r_shape}')

        # We just duplicate the end points to shift left/right
        full = np.stack([
            np.c_[resolutions[:, :1], resolutions].T,
            np.c_[resolutions, resolutions[:,-1:]].T,
        ], axis=0)

    # Full left/right resolution vectors already provided
    elif resolutions.ndim == 3: full = resolutions
    else: raise Exception(f'Max of 3 dimensions were expected: {r_shape}')

    # Shape should be (2, n_coordinates or 1, n_dimensions)
    assert(full.shape[0] == 2),            [full.shape, r_shape]
    assert(full.shape[1] in [1, n_coord]), [full.shape, n_coord]
    assert(full.shape[2] == n_dims),       [full.shape, n_dims]
    return full


class Grid:
    def __init__(self, 
        grid_index  : int | list[int],
        coordinates : np.ndarray | Callable,
        resolutions : np.ndarray | Callable,
        name  : None | str = None,
        dims  : None | list[str]  = None,
        table : None | np.ndarray = None, 
        ngrid : None | int = None,
    ):
        self.grid_index = list(np.atleast_1d(grid_index))
        self._C = coordinates
        self._R = resolutions
        self.name  = name or '+'.join([f'Grid_{i}' for i in self.grid_index])
        assert('datetime' not in self.name)
        self.dims  = dims or [f'Dim_{i}' for i in range(self.ndim)]
        self.table = np.arange(len(self))[:, None] if table is None else table
        self.ngrid = ngrid if ngrid is not None else len(self.grid_index)


    def __repr__(self) -> str:
        """ GridName(n_coordinates, n_dimensions) """
        return self.name#f'{self.name}({len(self)}, {self.ndim})'


    def __len__(self) -> int:
        """ Length is the number of coordinate points """
        return len(self.coordinates)


    def __eq__(self, other: 'Grid') -> bool:
        """ Two grids are equal if their coordinates and resolutions match """
        return ((len(self) == len(other)) and
                ((self.coordinates == other.coordinates) | (np.isnan(self.coordinates) & np.isnan(other.coordinates))).all() and
                ((self.resolutions == other.resolutions) | (np.isnan(self.resolutions) & np.isnan(other.resolutions))).all())


    def clone(self, **kwargs) -> 'Grid':
        """ Copy the current object, modifying any given parameters """
        return Grid(**{
            'grid_index'  : self.grid_index,
            'coordinates' : self.coordinates,
            'resolutions' : self.resolutions, 
            'name'        : self.name,
            'dims'        : self.dims,
            'table'       : self.table,
            'ngrid'       : self.ngrid,
        } | kwargs)


    def split(self) -> list:
        """ Split the current object into a list of composite subgrids """
        splits = {k: np.split(np.array(getattr(self, k)), self.ngrid, axis=-1)
                 for k in ['coordinates', 'resolutions', 'dims']}
        splits['dims'] = list(map(list, splits['dims']))
        return [self.clone(ngrid=1, **dict(zip(splits.keys(), v))) 
                for v in zip(*splits.values())]


    def combine(self, others: list):
        """ Combine a list of Grid objects into one """
        # print('others', others)
        # print([o.coordinates.shape for o in others])
        if len(others):
            return self.clone(**{
                # 'grid_index'  : sum([o.grid_index for o in others], []),
                'coordinates' : np.concatenate([o.coordinates for o in others], axis=-1),
                # 'coordinates' : np.c_[[o.coordinates for o in others]] if len(others) > 1 else others[0].coordinates,
                'resolutions' : np.dstack([o.resolutions for o in others]) if len(others) > 1 else others[0].resolutions, 
                'dims'        : sum([o.dims for o in others], []) if len(others) > 1 else others[0].dims,
                'ngrid'       : sum([o.ngrid for o in others]) if len(others) > 1 else others[0].ngrid,
            })
            # print('a',a.coordinates.shape, [o.coordinates.shape for o in others])
            # return a
        return self.clone(**{
            'coordinates' : self.coordinates[..., :0],
            'resolutions' : self.resolutions[..., :0], 
            'dims'        : [],
            'ngrid'       : 0,
        })
        # print('b', a.coordinates.shape)
        # return a 


    @property
    def ndim(self) -> int:
        """ Number of dimensions """
        return self.coordinates.shape[-1]

    @cached_property
    def virtual(self) -> np.ndarray:
        """ Boolean array indicating virtual dims (all elements are NaN) """
        return np.isnan(self.coordinates).all(0)

    @property
    def anisotropic(self) -> bool:
        """ Whether resolutions are anisotropic """
        return (self.resolutions.ndim > 1) and (self.resolutions.shape[1] > 1)

    @cached_property
    def coordinates(self) -> np.ndarray:
        """ Initialization coordinates can be callable for lazy evaluation """
        return self._C() if callable(self._C) else self._C

    @cached_property
    def resolutions(self) -> np.ndarray:
        """ Initialization resolutions can be callable for lazy evaluation """
        R = np.atleast_1d(self._R() if callable(self._R) else self._R)
        return full_resolutions(self.coordinates, R)


    def tiled_align(self, other: 'Grid') -> 'Grid':
        """ Tile this grid to match the other, if necessary """
        n_self  = self.ngrid#len(self.grid_index)
        n_other = other.ngrid#len(other.grid_index)
        if n_other > 1:# len(other.grid_index) > 1:

            #  First case: [a b] x3 -> [a b a b a b]
            # Second case: [a b] x3 -> [a a a b b b]
            case1 = lambda arr: np.tile(arr, n_other)
            split = lambda arr: map(case1, np.split(arr, n_self, axis=-1))
            case2 = lambda arr: np.concatenate(list(split(arr)), axis=-1)
            
            # Use the first index to break symmetry
            tile = case1 if self.grid_index[0] < other.grid_index[0] else case2

            return self.clone(**{
                'coordinates' : tile(self.coordinates),
                'resolutions' : tile(self.resolutions),
                'dims'        : list(tile(np.array(self.dims))),
            })
        return self


    def reorder(self, col_order: list[int]) -> (np.ndarray, 'Grid'):
        """ Create a new grid with reordered columns,
            ensuring lexicographic ordering of rows """
        row_order = np.lexsort(self.coordinates[..., col_order].T[::-1])
        return row_order, self.clone(**{
            'coordinates' : self.coordinates[..., col_order][row_order],
            'resolutions' : self.resolutions[..., col_order][:, row_order]
                if self.anisotropic else self.resolutions[..., col_order],
        })


class Pair:
    def __init__(self, G1: Grid, G2: Grid):
        self.G1 = G1
        self.G2 = G2
        self.i1, self.i2 = self._get_indices()


    def _get_indices(self, optimize=True, debug=True):
        # Duplicate grids as necessary to align with each other
        G1 = self.G1.tiled_align(self.G2)
        G2 = self.G2.tiled_align(self.G1)
        assert(G1.ndim == G2.ndim), [G1.ndim, G2.ndim]
        assert(G1.dims == G2.dims), [G1.dims, G2.dims]

        # Find any all-NaN (virtual) columns in the grids
        skip_dims = G1.virtual | G2.virtual

        # If optimizing, swap grids as necessary to use optimal order
        if optimize:

            # Check virtual dims count (i.e. all values in column are NaN):
            #  Grid with fewer samples is first when same number of virtuals
            #  Otherwise, grid with more virtual dims is first
            v1, v2 = G1.virtual.sum(), G2.virtual.sum()
            switch = (len(G1)>len(G2)) if v1==v2 else (v1<v2)
            if debug: print(f'\t{v1=} {v2=} {switch=}')
        else: switch = False
        
        order = self.order = slice(None, None, -1 if switch else 1)
        G1,G2 = [G1, G2][order] 

        # If optimizing, reorder the columns as necessary to use optimal order
        if optimize:

            # Order columns by the first grid column's entropy values
            cols = np.argsort([ np.inf if skip else entropy(col)[0] for skip, col
                                in zip(skip_dims, G1.coordinates.T) ])
            row_order_1, G1 = G1.reorder(cols)
            row_order_2, G2 = G2.reorder(cols)
            skip_dims = skip_dims[cols]

        if debug:
            print('  b q shape:', G1.coordinates.shape, G2.coordinates.shape)
            print('br qr shape:', G1.resolutions.shape, G2.resolutions.shape)
            print('  skip dims:', skip_dims)

        # if G1.coordinates.shape == (9539470, 18):
        #     from crest.data.loading import Dataset
        #     Dataset.interactive()
        # Perform the actual matchup procedure
        idxs_1, idxs_2 = bruteforce_double(
            G1.coordinates, G2.coordinates, 
            *G1.resolutions, *G2.resolutions, 
            skip_dims, None,
        )[order]
        print('finish:', idxs_1.shape, idxs_2.shape)
        # If no matches are found, we can immediately return 
        if min(idxs_1.size, idxs_2.size) == 0: 
            return (np.empty((0, 0)),) * 2

        # Recover the original row indices
        if optimize:
            row_order_1, row_order_2 = [row_order_1, row_order_2][order]
            idxs_1 = row_order_1[idxs_1]
            idxs_2 = row_order_2[idxs_2]
        return idxs_1, idxs_2


    @property
    def table(self):
        return np.c_[(self.G2.table[self.i2], self.G1.table[self.i1])[::-1][self.order]]

    @property
    def grid_order(self):
        return sum([self.G2.grid_index, self.G1.grid_index][::-1][self.order], [])

    @property
    def dims(self):
        return sum([self.G2.dims, self.G1.dims][::-1][self.order], [])

    @property
    def name(self):
        return '+'.join(map(str, [self.G2.name, self.G1.name][::-1][self.order]))


    def coordinates(self):
        # print('\n\nHERE')
        # print(self.G2.coordinates.shape, self.i2.shape)
        # print(self.G1.coordinates.shape, self.i1.shape)
        # print('----------------------------------------------\n\n')
        return np.c_[(
                self.G2.coordinates[self.i2],
                self.G1.coordinates[self.i1],
            )[::-1][self.order]]


    def resolutions(self):
        R1, R2 = self.G1.resolutions, self.G2.resolutions

        # If either grid uses a non-uniform resolution, both need to
        if self.G1.anisotropic or self.G2.anisotropic:
            if R1.shape[1] == 1: R1 = np.tile(R1, (1, len(self.G1), 1))
            if R2.shape[1] == 1: R2 = np.tile(R2, (1, len(self.G2), 1))
            R12 = [R1[:, self.i1], R2[:, self.i2]]
        else: R12 = [R1, R2]
        return np.dstack(R12[self.order])

    def combined_grid(self, check_duplication: bool = True):
        # Construct the next query set by combining the current two grids
        if check_duplication:
            def update(n):
                G = getattr(self, f'G{n}')
                i = getattr(self, f'i{n}')
                R = G.resolutions

                # If either grid uses a non-uniform resolution, both need to
                if self.G1.anisotropic or self.G2.anisotropic:
                    if R.shape[1] == 1: R = np.tile(R, (1, len(G), 1))
                    R = R[:, i]
                return G.clone(**{
                    'coordinates' : G.coordinates[i],
                    'resolutions' : R, 
                })

            # q = self.coordinates()
            # print(f'current: {q[0]} {q.shape}')
            # print('G1', self.G1.coordinates[self.i1][0], self.G1.coordinates[self.i1].shape)
            # print('G2', self.G2.coordinates[self.i2][0], self.G2.coordinates[self.i2].shape)

            G1, G2 = [2, 1][::-1][self.order]
            G1s = update(G1).split()
            G2s = update(G2).split()
            new = []
            for i,g1 in enumerate(G1s):
                for j,g2 in enumerate(G2s):
                    if g1 == g2: 
                        # print(f'{g1} == {g2}')
                        break
                    else:        new.append(i)
            orig = getattr(self, f'G{G1}').split()
            setattr(self, f'G{G1}', getattr(self, f'G{G1}').combine([orig[i] for i in new]))

            # print('\nTHIS')
            # print(self.G1.coordinates.shape)
            # print(self.G2.coordinates.shape)
            # q = self.coordinates()
            # print(f'New: {q[0]} {q.shape}')
            # print('G1', self.G1.coordinates[self.i1][0], self.G1.coordinates[self.i1].shape)
            # print('G2', self.G2.coordinates[self.i2][0], self.G2.coordinates[self.i2].shape)

            # C = np.split(self.coordinates(), len(self.grid_order), axis=-1)
            # R = np.split(self.resolutions(), len(self.grid_order), axis=-1)
            # G = [self.G2, self.G1][::-1][self.order][0]
            # n1 = len(G.grid_index)
            # C1, C2 = C[:n1], C[n1:]
            # R1, R2 = R[:n1], R[n1:]
            # dup = []
            # for i, (c1, r1) in enumerate(zip(C1, R1)):
            #     for j, (c2, r2) in enumerate(zip(C2, R2)):
            #         if (c2 == c1).all() and (r2 == r1).all():
            #             dup.append(i)
            #             break
            # if dup:
            #     print(f'Discarding {dup} from {G}  {G.coordinates.shape}  {G.resolutions.shape}')
            #     print(G.coordinates[0])
            #     print(G.resolutions[:,0])
            #     def cat(arr):
            #         new = [v for i,v in enumerate(np.split(np.array(arr), len(G.grid_index), axis=-1)) if i not in dup]
            #         if new: return np.concatenate(new, axis=-1)
            #         return np.array(arr)[..., :0]
            #     G.coordinates = cat(G.coordinates)
            #     G.resolutions = cat(G.resolutions)
            #     G.dims = list(cat(G.dims))
            #     G.grid_index = list(cat(G.grid_index))
            #     print(f'New G: {G}  {G.coordinates.shape}  {G.resolutions.shape}')
            #     print(G.coordinates[0])
            #     print(G.resolutions[:,0])
            # for i, c in enumerate(C):
            #     if (last == c).all():
            #         *R, last = np.split(self.resolutions(), len(self.grid_order), axis=-1)
            #         if (last == R[i]).all():
            #             keep = [self.G2, self.G1][self.order][0]
            #             print(f'=== Duplicated grids in {self.name}: -1 == {i}')
            #             print(f'  Keeping {keep}')

            #             return Grid(**{
            #                 'grid_index'  : keep.grid_index,
            #                 'coordinates' : keep.coordinates,
            #                 'resolutions' : keep.resolutions, 
            #                 'name'        : self.name,
            #                 'dims'        : keep.dims,
            #                 'table'       : self.table,
            #             })
          
        return Grid(**{
            'grid_index'  : self.grid_order,
            'coordinates' : self.coordinates(),
            'resolutions' : self.resolutions, 
            'name'        : self.name,
            'dims'        : self.dims,
            'table'       : self.table,
            'ngrid'       : self.G1.ngrid + self.G2.ngrid,
        })



def find_neighbors2(
    coordinates : Collection[np.ndarray], 
    resolutions : Collection[np.ndarray] | None = None,
    radius      : float = 0.5,
    method      : str   = 'brute',
    allow_empty : bool  = False,
    use_implode : bool  = False,
    shuffle     : bool  = False,
    debug       : bool  = False,
    logger      : logging.Logger | None = None,
    eps         : float = 1e-5,
    grid_labels : Collection[str] | None = None,
    axis_labels : Collection[str] | None = None,
    **kwargs,
) -> (np.ndarray, np.ndarray):
    """ Find all nearest neighbors for the given coordinates.

    Notes
    -----
    Ordinarily, finding nearest neighbors for multiple sets of coordinate 
    grids would have combinatorial time complexity. This function implements
    a variety of methods which are able to utilize structural information 
    inherent to the grids, in order to reduce the time complexity to 
    O(K*Dlog(N)); where K is the number of grids, D is the dimensionality 
    of the grids, and N is the size of the grids. By default, neighbor
    distances are calculated based on the L-infinity norm, with the given
    radius used as a fraction of the grids' coordinate resolution (e.g.
    radius=0.5 means for a given grid point, any queried points which fall
    within 1/2 the resolution of that grid's coordinates around the given
    point, would be regarded as a neighbor).

    Parameters
    ----------
    coordinates : Collection[np.ndarray]
        Coordinate locations to find neighbors for.
    resolutions : Collection[np.ndarray] | None
        Resolution of the given coordinate locations, which is used
        to normalize the coordinates prior to building the BallTrees.
        If it is not given, a resolution of 1 is used for all dimensions.
        For the `brute` method, resolutions can be a 3d matrix of shape
        (N, D, 2) - which represents the number of coordinate grids, the
        dimensionality, and the left and right hand resolution, respectively.
        This allows non-uniform coordinate spacing, including anisotropic 
        coordinate systems. 
    radius      : float
        Radius within which points are considered neighbors of a reference point
        (with radius indicating a fraction of the resolution).
    method      : str
        Method to use for finding neighbors. Options are:

        - brute (default)
            Brute force neighbor search which exploits monotonically increasing
            coordinate grids. Note that coordinates passed in *must* be 
            lexicographically sorted within each grid's coordinates, but 
            generally in CREST this comes for free where finding neighbors 
            happens. This brute force approach is the only implemented method 
            which allows grids with non-uniform spacing (rectilinear grids, as
            well as anisotropic coordinate systems) to be accurately matched. 
            It should also be at least on par with the speed of other comparable 
            methods, if not faster - esp. when more grids and/or number of 
            dimensions are used. Further (relatively significant) optimizations
            could be made to the method as well, if warranted. Only allows 
            L-infinity norm to be used in checking neighbor distances. 
        - tree
            Iteratively build up the matches by combining grids together and
            extending the coordinate dimensionality. A KDTree is constructed 
            on each iteration, with the output of one tree being fed into the
            next such that the coordinates are combined together. This method
            provides simultaneous grid matching, and can sometimes be faster
            than `brute` if coordinates are regular grids (uniform spacing). 
        - polars
            Use a table inner join to find matches. Slower than using a single
            anchor, but ensures any returned matches are simultaneously matched
            across all grids; i.e. [i,j,k] implies A[i] == B[j] == C[k]. Note 
            that order of returned matches can sometimes vary. 
        - pandas
            Same as `polars`, but uses the pandas library instead of polars.
            In general, `polars` should be preferred since it will produce
            the same result but operate faster than using `pandas`. 
        - anchor
            Builds N-1 BallTrees and queries each of them using a reference 
            coordinate set (where N=len(coordinates)). The reference 
            coordinate set is the first index in the given collection of 
            coordinates. Generally much faster than methods which ensure 
            matches are simultaneous across all grids, but also returns 
            many matches which may not be considered actual neighbors; 
            i.e. [i,j,k] implies A[i] == B[j] and A[i] == C[k], but not 
            necessarily B[j] == C[k].

    allow_empty : bool
        Whether to allow empty neighbor sets in the results (i.e. all reference
        points are returned, regardless of if there are any neighbors). By
        default, only reference points which have at least one neighbor in 
        all other coordinate sets are returned. Note that this option is only 
        available when `use_anchor=True`; otherwise has no effect.
    use_implode : bool
        Condense matches into nested lists rather than returning a flattened
        representation. Takes slightly longer to run, but can significantly
        reduce memory requirements in some cases. See the `implode` method 
        for further details.
    shuffle : bool
        If True, shuffle the table before returning (only applicable for
        method='brute').
    eps     : float
        Small constant added for numerical stability.
    **kwargs
        Additional keywords are passed to sklearn.neighbors.BallTree.

    Returns
    -------
    (np.ndarray, np.ndarray)
        A tuple of two arrays: neighbor indices, and neighbor counts.
        Both arrays are shaped [len(coordinates), len(coordinates[0])],
        but neighbor indices (first array) is a ragged object array where 
        each element is itself a variable length array::

          [ 
             array(ref grid indices) 
             array([grid 2 indices matching ref grid indices[0]], 
                   [grid 2 indices matching ref grid indices[1]],
                   [...])
             array([grid 3 indices matching ref grid indices[0]],
                   [grid 3 indices matching ref grid indices[1]],
                   [...])
          ]

        For example, indices equal to 

            [ [[1], [2]],  [[0, 1], [2]],  [[2], [4,5]] ]
        
        would indicate:
        
        - ref_grid[1] matches [grid_2[0], grid_2[1]] and [grid_3[2]]
        - ref_grid[2] matches [grid_2[2]] and [grid_3[4], grid_3[5]]
        
        Counts (the second array output) is then the length (number of 
        matches) for each coordinate grid neighbor set, i.e. the length
        of each nested ragged array. Using the above example, counts would
        correspond to::
        
            [[1, 1], [2, 1], [1, 2]]
        
        as the matches for the ref grid have lengths 1 and 1; for grid_2 
        have lengths 2 and 1; and for grid_3 have lengths 1 and 2. 

    """
    # import pickle as pkl
    # with open('neighbors_samegrids.pkl', 'wb') as f:
    #     pkl.dump([coordinates, resolutions, grid_labels, axis_labels], f)
    # assert(0)
    # Rough guess on what dtype can be used to hold grids / indices
    large = max(map(np.log10, map(len, coordinates))) > 8
    itype = np.int64 if large else np.int32
    isflt = lambda v: np.issubdtype(v, np.floating) 
    ftype = max(filter(isflt, [c.dtype for c in coordinates]+[np.float32]))

    if resolutions is not None:
        # Sanity check
        for c, r in zip(coordinates, resolutions):
            r = np.atleast_1d(r)
            if r.ndim == 3: r = r[..., 0]
            if r.ndim == 2: r = r.max(0)
            if ((np.abs(c).max(0) > 0) & (r > np.abs(c).max(0))).any():
                raise Exception(f'Resolution > Coordinate: {r} > {c.max(0)}')

    # Set a default value for the resolutions / labels if None was given
    resolutions = resolutions or [np.ones(c.shape[-1]) for c in coordinates]
    grid_labels = grid_labels or [f'Grid_{i}' for i in range(len(coordinates))]
    axis_labels = axis_labels or [[f'Dim_{i}' for i in range(c.shape[-1])] for c in coordinates]

    if len(coordinates) == 1:
        table = np.arange(len(coordinates[0]), dtype=itype)[None]
        count = np.ones_like(table)
        return table, count 

    # Create the grid objects
    format_res = lambda r: (np.moveaxis(r, -1, 0) if r.ndim == 3 else r) * radius + eps
    keys, vals = zip(*{
        'coordinates' : [c.astype(ftype) for c in coordinates], 
        'resolutions' : [format_res(r).astype(ftype) for r in resolutions], 
        'name'        : grid_labels, 
        'dims'        : axis_labels,
    }.items())
    grids = [Grid(i, **dict(zip(keys, v))) for i, v in enumerate(zip(*vals))]

    # Optimize column and grid orderings
    optimizations = True
    if optimizations: 
        make_sizes = lambda G: float(f'{G.resolutions.size}.{G.coordinates.size}')
        grid_order = np.argsort(list(map(make_sizes, grids)))[::-1]
    else: grid_order = list(range(len(grids)))
    grids = [grids[i] for i in grid_order]

    # Create the grids to iterate over and pull out the first query

    if debug:
        print('\nShapes:')
        with print_table(['Grid', 'Coordinates', 'Resolutions']) as printer:
            for G in grids: 
                printer(G, G.coordinates.shape, G.resolutions.shape)

        for i, (label, values) in enumerate([
            ('Coordinate', [G.coordinates    for G in grids]), 
            ('Resolution', [G.resolutions[0] for G in grids]),
        ]):
            print(f'\n{label} Extents:')
            with print_table(['Grid', 'Axis', 'Minim', 'Maxim']) as printer:
                for G, v in zip(grids, values):
                    for d, minim, maxim in zip(G.dims, v.min(0), v.max(0)):
                        if (d == 'datetime') and np.isfinite(v).all():
                            otype = np.timedelta64 if i else np.datetime64
                            minim = otype(int(minim), 'm')
                            maxim = otype(int(maxim), 'm')
                        printer(G, d, minim, maxim)
                        G = ''

        print(f'\nStarting neighbor search with {grids[0]}: {len(grids[0]):,} possible matches')

    if logger is not None:
        logger.debug(f'\tStarting neighbor search with {grids[0]}: {len(grids[0]):,} possible matches')

    # grids = [grids[i] for i in grid_order]
    # prime = grids.pop(0)
    # grids = grids[::-1]

    def select_pair(grids):
        print('\nSelecting pair from:')
        with print_table(['Grid', 'Coordinates', 'Resolutions']) as printer:
            for G in grids: 
                printer(G, G.coordinates.shape, G.resolutions.shape)

        # if len(grids) == 6: return grids.pop(-1), grids.pop(-1)
        # if len(grids) == 5: return grids.pop(-1), grids.pop(-1)
        # if len(grids) == 4: return grids.pop(-1), grids.pop(-1)
        # assert(0)
        # # First check for grids that are from the same source
        import re
        pattern = re.compile(r'Block\[\d+\:(.*)_.+\]')
        sources = [(pattern.findall(G.name) + [None])[0] for G in grids]
        isvalid = lambda s: (s is not None) and (sources.count(s) > 1)
        options = list(filter(isvalid, set(sources)))
        if options:
            ix1 = sources.index(options[0])
            ix2 = sources.index(options[0], ix1+1)
            print(f'[Sources] Selected indices {ix1} and {ix2}')
            return grids.pop(ix2), grids.pop(ix1)

        # Next check if there are any uniform resolution grids
        uniform = [not G.anisotropic for G in grids]
        if sum(uniform) > 1:
            ix1 = uniform.index(True)
            ix2 = uniform.index(True, ix1+1)
            print(f'[Uniform] Selected indices {ix1} and {ix2}')
            return grids.pop(ix2), grids.pop(ix1)

        # Otherwise, just select the first two grids
        print(f'[Default] Selected indices -1 and -1')
        return grids.pop(-1), grids.pop(-1)


    for i in range(1, len(grids)):
        G1, G2 = select_pair(grids)
        # G1 = grids.pop(-1)
        # G2 = grids.pop(-1)

        iter_timer = Stopwatch(f'Iteration {i}/{len(grid_order)-1}')
        iter_timer.__enter__()
        print(f'\n{iter_timer}')
        print(''.join(['-']*len(str(iter_timer))))
        print('Query label:', G2.name)
        print('Build label:', G1.name)
        print('Query shape:', G2.coordinates.shape, G2.resolutions.shape)
        print('Build shape:', G1.coordinates.shape, G1.resolutions.shape)
        print('Table shape:', G1.table.shape, G1.table.dtype)
        print('Grid1 param:', G1.name, G1.dims, G1.ngrid, G1.grid_index)
        print('Grid2 param:', G2.name, G2.dims, G2.ngrid, G2.grid_index)

        pair = Pair(G1, G2)
        if pair.i1.size == 0:
            return (np.empty((0, 0)),) * 2
        grids.append( pair.combined_grid() )

        # g = grids[-1]
        # print(g, g.coordinates.shape, g.resolutions.shape, g.grid_index)
        # iter_timer.__exit__()
        # c1, c2 = np.split(g.coordinates, 2, axis=-1)
        # print(c1.shape, c2.shape)
        # print((c1 == c2).all())
        # r1, r2 = np.split(g.resolutions, 2, axis=-1)
        # print(r1.shape, r2.shape)
        # print((r1 == r2).all())
        iter_timer.__exit__()
        # assert(0)

        print()
    table = grids[0].table
    order = grids[0].grid_index
    print(order)

    # Reorder the columns correctly 
    if debug: print('\nReordering table...')
    table = table[:, np.argsort(order)[np.argsort(grid_order)]]

    # Random sample ordering
    if shuffle:
        if debug: print('Shuffling...')
        i = np.arange(len(table))
        np.random.shuffle(i)
        table = table[i].T

    # Lexigraphic sort to have consistent return order
    else:
        if debug: print('Lexsorting table...')
        # lexsort(table); table = table.T
        table = table[np.lexsort(table.T[::-1])].T

    if debug: print('Finishing...')
    if use_implode:
        table = implode(table)
        count = np.array([list(map(len, col)) for col in table], dtype=itype)
    else:
        count = np.ones_like(table, dtype=itype)

    return table, count



class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    Source: https://stackoverflow.com/a/36296215/22210498
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                line = line.strip()
                if line:
                    self.logger.log(self.log_level, line.strip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            line, self.linebuf = self.linebuf.strip(), ''
            if not (line.startswith('0%') or line.startswith('100%')):
                self.logger.log(self.log_level, line)
        