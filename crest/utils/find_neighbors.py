from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from collections.abc import Collection
from contextlib import nullcontext
from itertools import combinations, product, chain, starmap
from functools import reduce, partial
from threading import Timer

# Allow bruteforce progress logging if numba_progress available
try:                from numba_progress import ProgressBar
except ImportError: ProgressBar = None 

try:                import polars as pl
except ImportError: pl = None
    
import numpy as np
import pandas as pd 
import logging 

# from ._bruteforce import *
from .print_table import print_table
from .Stopwatch import Stopwatch
from crest.utils.matchup.bruteforce.utils.entropy import entropy  
from crest.utils.matchup.bruteforce import brute

t = Timer(5, lambda: print('Compiling numba functions...'))
t.start()
# from .lexsort import lexsort
from .matchup.bruteforce.utils.bruteforce_numba import *
from .matchup.bruteforce.utils.multiset_numba import multiset_single_numba
if t.is_alive():
    t.cancel()
else: print('Finished compiling functions')
    

def get_indices(
    coordinates : Collection[np.ndarray],
    resolutions : Collection,
    radius      : float,
    dtype       : type = np.int32,
    expand      : bool = False,
    balltree    : bool = False,
    use_faiss   : bool = False,
    **kwargs
) -> Collection:
    """Query a KD/BallTree to find neighbors.

    Parameters
    ----------
    coordinates : Collection[np.ndarray]
        A collection of exactly two arrays - [`build`, `query`] - where
        `build` is used to construct the tree, and `query` is used to 
        query it.
    resolutions : Collection
        A collection of exactly two objects, representing the resolution
        (i.e. normalization factor) for each of the given coordinate 
        arrays. Note that the order of this collection does not matter.
    radius      : float
        Max distance to consider a point a neighbor. 
    dtype       : type
        dtype for the expanded index arrays.
    expand      : bool
        Whether the neighbor indices should be expanded, or remain in its
        condensed representation; see the Returns section for discussion.
    balltree    : bool
        If True, use `sklearn.neighbors.BallTree` to find neighbors; use
        `scipy.spatial.KDTree` otherwise. Note that BallTree returns an
        array of arrays, while KDTree returns an array of lists; this 
        usually doesn't matter when using the results, but may have 
        implications for function speed/memory/etc.  
    use_faiss   : bool
        Use FAISS to find neighbors. Performs neighbor search faster, and 
        with a much smaller memory footprint than other methods. Also allows
        for both exact and approximate searches, with a significant runtime 
        reduction if finding only a subset of all matches is reasonable. Note 
        that this method is still under development.
    **kwargs
        Any other parameters to pass to the tree class/query function.

    Returns
    -------
    np.ndarray | (np.ndarray, np.ndarray)
        If `expand` is False, the neighbor representation is returned in
        its condensed form; this returned array will be a ragged object 
        array of length `len(coordinates[1])` (i.e. `len(query)`), where 
        each element is a collection (list or array) containing indices 
        of `coordinates[0]` (i.e. `build`) that match the element's index. 
        For instance, the return array `[[2], [], [1,4]]` would indicate:

        - `query[0]` matches `build[2]`
        - `query[1]` does not match any `build` elements
        - `query[2]` matches `build[1]` and `build[4]`

        Note that 'matches' in this context means that two points are
        within a distance of `radius` from each other, with the distance
        metric dependent on any `kwargs` given (L2 by default).

        If `expand` is True, returns the un-nested indices for both arrays. 
        Using the example above, instead of `[[2], [], [1,4]]`, the returned 
        value would be (`[2, 1, 4]`, `[0, 2, 2]`).

    """

    def concat(array, dtype, count=-1):
        """ Faster version of np.concatenate that does not cause thread
            contention (see https://github.com/numpy/numpy/issues/24252) """
        return np.fromiter((a[i] for a in array for i in range(len(a))), dtype, count)

    def mask(coordinates: Collection[np.ndarray]) -> Collection[np.ndarray]:
        """Remove axes which contain all nan values.
        
        The standard use of this would be to pass two grids that are being 
        used to build/query a tree, so that both have the same axes masked.

        Parameters
        ----------
        coordinates: Collection[np.ndarray]
            Any number of coordinate grids. 

        Returns
        -------
        Collection[np.ndarray]
            A collection of the same `coordinates` that were given as input,
            with all coordinate arrays having the same axes masked (if any).
            The masking process checks that if any values along an axis are
            NaN, that all values along that axis are NaN; any axes which are
            found to contain solely NaN values are masked in all arrays.

        """
        def nan_axis(c):
            """ If an axis contains nan values, all axis values must be nan """
            nans = ~np.isfinite(c) # Also check non-finite values such as inf
            some = nans.any(0)
            assert(np.where(some, nans.all(0), True).all())
            return np.where(some)[0].tolist()

        # Get a set of all invalid axes, then return the filtered arrays
        invalid = set( sum(map(nan_axis, coordinates), []) ) 
        make_2d = lambda c: c[:, None] if c.ndim < 2 else c
        rm_axes = lambda c: c[:, tuple(set(range(c.shape[1])) - invalid)]
        return tuple(map(rm_axes, map(make_2d, coordinates)))

    def query_tree(build, query):
        if use_faiss:
            import faiss 
            faiss.omp_set_num_threads(1) # Minimize dask thread contention

            # with Stopwatch('build index'):
            # FAISS crashes when Linf metric is used (or any other non-default metric)
            nfeat = build.shape[1]
            nlist = int(max(100, 0.5 * (len(build) ** 0.5)))
            quant = faiss.IndexFlatL2(nfeat)#, faiss.METRIC_Linf)
            index = faiss.IndexIVFFlat(quant, nfeat, nlist)#, faiss.METRIC_Linf)
            param = faiss.SearchParametersIVF(nprobe=5)

            index.train(build)
            index.add(build)

            # with Stopwatch('range search'):
            lims, dist, idxs = index.range_search(query, radius, params=param)

            query_i = np.repeat(np.arange(len(query), dtype=dtype), np.diff(lims.astype(dtype)))
            build_i = idxs
            return build_i, query_i

        if balltree:
            return BallTree(build,  **kwargs).query_radius(query, radius)

        kd_kwargs = {
            'compact_nodes' : False,
            'balanced_tree' : False,
            'leafsize'      : 100,
        }
        kwargs.update({'return_sorted': False})
        return KDTree(build, **kd_kwargs).query_ball_point(query, radius, **kwargs)

    maxim = np.nanmax(resolutions, axis=0, keepdims=True)
    shape = lambda c: c[:, None] if c.ndim < 2 else c
    scale = lambda c: c / maxim

    # Normalize coordinates and find neighbors within the requested radius
    processed = mask( list(map(scale, map(shape, coordinates))) )
    # print('maxim', maxim)
    # print('processed')
    # for p in processed:
    #     print(p)
    #     print()

    neighbors = query_tree(*processed)
    if use_faiss: return neighbors

    # Expand the neighbor index representation; see docstring for discussion
    if expand:
        lengths = list(map(len, neighbors)) 
        build_i = concat(neighbors, dtype=dtype, count=sum(lengths))
        query_i = np.repeat(np.arange(len(lengths), dtype=dtype), lengths)
        assert(len(lengths) == len(coordinates[1])), [len(lengths), [len(c) for c in coordinates], len(build_i)]
        return build_i, query_i
    return neighbors


def implode(table):#: np.ndarray | pl.DataFrame):
    """Inverse of DataFrame.explode.

    Takes a row-expanded representation of data and transforms it into
    a condensed ragged array. For example:

    >>> a = [ [0, 0, 0],
    ...       [0, 0, 1],
    ...       [0, 1, 0],
    ...       [1, 1, 1] ]
    >>> implode(a.T)
    [ [[0], [0], [0,1]],
      [[0], [1], [0]  ],
      [[1], [1], [1]  ] ]

    """
    if pl is None: 
        raise ImportError('implode requires polars to be installed')
        
    # Exclude one name from the full list
    # Groupby all names except one, then concat into a comma-delimited string
    excl = lambda remove: list( set(names) - {remove} )
    join = lambda tbl, n: tbl.group_by(excl(n)).agg(pl.col(n).str.concat(','))

    if isinstance(table, (pl.DataFrame, pl.LazyFrame)):
        dtype = table.dtypes[0]
        names = table.columns 
    else:
        # assert(table.shape[0] <= table.shape[1]), f'table is likely transposed: {table.shape}'
        dtype = getattr(pl, table.dtype.name.title())
        names = [f'col_{i}' for i in range(len(table))]
        table = pl.DataFrame(dict(zip(names, table))).lazy()

    # Cast all columns to string
    table = table.with_columns(pl.col(names).cast(str))

    # Apply join in reverse to all but the first column;
    # then sort by the first column, parse all columns  
    # into lists of ints, restore original column order,
    # and finally collect and return as a numpy array
    try:
        return ( reduce(join, names[::-1], table)
            .sort(names)
            .with_columns(
                pl.col(names).str.split(',')
                  .cast(pl.List(dtype)))
            .select(names)
            .collect().to_numpy().T # Ensure pyarrow installed if error
        )
    except: 
        print('Ensure pyarrow is installed.')
        raise


def multiset_single(
    coordinates : list[np.ndarray], 
    resolutions : list[np.ndarray], 
    num_samples : int = -1,
    shuffle     : bool = True,
    logger      : logging.Logger | None = None, 
):
    log = getattr(logger, 'debug', print)
    with Stopwatch('multiset_single preparation', log, silent=logger is None):
        # Temporarily hard-code reversing feature dimension in order to
        # put the datetime coordinate last
        orig_idxs = []
        for i in range(len(coordinates)):
            coordinates[i] = coordinates[i][:, ::-1]
            resolutions[i] = resolutions[i][:, ::-1]
            idx = np.lexsort(coordinates[i].T[::-1])
            orig_idxs.append(np.arange(len(coordinates[i]))[idx])
            coordinates[i] = coordinates[i][idx]
            if len(resolutions[i]) > 1:
                resolutions[i] = resolutions[i][idx]
        
        array_lens = np.array([len(a) for a in coordinates], dtype=np.int32)

        order = np.argsort(array_lens)
        array_lens = array_lens[order].astype(np.int32)
        array_idxs = np.cumsum(np.r_[[0], array_lens])
        array_idxs = np.stack([array_idxs[:-1], array_idxs[1:]], axis=1).astype(np.int32)

        c = np.vstack([coordinates[i] for i in order])
        for i in range(len(coordinates)):
            coordinates[i] = None
        coordinates = None
        c = np.tile(c, (1, 3))

        for i, j in enumerate(order):
            r = resolutions[j]
            c[array_idxs[i, 0]:array_idxs[i, 1], r.shape[1]:] += np.c_[-r[..., 0], r[..., 1]]
            resolutions[j] = r = None
        resolutions = None
        if logger is not None:
            log(f'{array_lens=} {order=}')

        n_threads = nb.get_num_threads()
        chunksize = 0
        if array_lens[0] > (n_threads*2):
            chunksize = 1+array_lens[0] // (n_threads * 8)
        # nb.set_num_threads(n_threads)
        nb.set_parallel_chunksize(chunksize) 
        
        if logger is not None:
            log(f'Set numba {chunksize=} (num threads={n_threads})')
        
    table = multiset_single_numba(c, array_lens, array_idxs, num_samples, shuffle)
    order = np.argsort(order)
    return np.array([i[t] for t,i in zip(table.T[order], orig_idxs)]).T

    
def find_neighbors(
    coordinates : Collection[np.ndarray], 
    resolutions : Collection[np.ndarray] | None = None,
    radius      : float = 0.5,
    method      : str   = 'multi',
    allow_empty : bool  = False,
    use_implode : bool  = False,
    shuffle     : bool  = False,
    debug       : bool  = False,
    logger      : logging.Logger | None = None,
    eps         : float = 1e-5,
    num_samples : int   = -1,
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

    # Rough guess on what dtype can be used to hold indices and values
    large = max(map(np.log10, map(len, coordinates))) > 8
    itype = np.int64 if large else np.int32
    isflt = lambda v: np.issubdtype(v, np.floating) 
    ftype = np.float32#max(filter(isflt, [c.dtype for c in coordinates]+[np.float32]))

    # If there's only one grid, we can just return the indices for it
    if len(coordinates) == 1:
        table = np.arange(len(coordinates[0]), dtype=itype)[None]
        count = np.ones_like(table)
        return table, count 

    # Add small value to radius to account for numerical instability
    # Bear in mind this is related to the resolution's rounding
    kwargs.update({
        'radius' : radius + eps,
        'dtype'  : itype,
        'p'      : kwargs.get('p', np.inf),
    })

    # Set the radius, dtype, and any other kwargs given
    get_matches = partial(get_indices, **kwargs)

    # if resolutions is not None:
    #     # Sanity checks
    #     for c, r in zip(coordinates, resolutions):
    #         r = np.atleast_1d(r)
    #         if r.ndim >  3: raise Exception(f'Found resolution ndim > 3: {r.shape}')
    #         if r.ndim == 3: r = r[..., 0]
    #         if r.ndim == 2: r = r.max(0)
    #         if ((np.abs(c).max(0) > 0) & (r > np.abs(c).max(0))).any():
    #             raise Exception(f'Resolution > Coordinate: {r} > {c.max(0)}')

    # Set a default value for the resolutions if None was given
    resolutions = resolutions or [np.ones(c.shape[-1]) for c in coordinates]
    grid_labels = grid_labels or [f'Grid_{i}' for i in range(len(coordinates))]
    axis_labels = axis_labels or [[f'Dim_{i}' for i in range(c.shape[-1])] for c in coordinates]

    if not hasattr(coordinates, '__getitem__'): coordinates = list(coordinates)
    if not hasattr(resolutions, '__getitem__'): resolutions = list(resolutions)

    # Update dtypes
    for i in range(len(coordinates)):
        c = coordinates[i]
        r = resolutions[i]

        if not isinstance(c, np.ndarray) or (c.ndim < 1):
            coordinates[i] = c = np.atleast_1d(c)
        if not isinstance(r, np.ndarray) or (r.ndim < 1):
            resolutions[i] = r = np.atleast_1d(r) 
        if c.dtype != ftype:
            coordinates[i] = c.astype(ftype)
        if r.dtype != ftype:
            resolutions[i] = r.astype(ftype)

    # For anisotropic grids, the only available method is bruteforce
    if any(r.ndim > 2 for r in resolutions): method = 'multi'

    # Single anchor grid, checked against all other grids
    if method == 'anchor':
        # Create trees and query against the reference (anchor) grid
        coordinates = product(coordinates[1:], coordinates[:1])
        resolutions = product(resolutions[1:], resolutions[:1])
        match = list(map(get_matches, coordinates, resolutions))
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

    # [Multiple anchors] Inner join over all combinations of grids
    if method in ['polars', 'pandas']:

        def build_frame(keys, *cs_rs):
            """ Create a dataframe with the names/coordinates/resolutions """
            build_i, query_i = get_matches(*cs_rs, expand=True)
            if method == 'polars':
                if pl is None:
                    raise ImportError(f'polars must be installed for {method=}')
                return pl.DataFrame(dict(zip(keys, [build_i, query_i]))).set_sorted(keys[1]).lazy()
            return pd.MultiIndex.from_arrays([build_i, query_i], names=keys)

        # Perform an inner join on matches over all combinations of grids
        indices = list(range(len(coordinates)))
        columns = [f'col_{i}' for i in indices]
        pairing = partial(combinations, r=2)
        frames  = map(build_frame, *map(pairing, [columns, coordinates, resolutions]))

        c_set = lambda df1, df2: list( set(df2.columns) & set(df1.columns) )
        join  = lambda df1, df2: df1.join(df2, how='inner', **({} if method=='pandas' else {'on': c_set(df1, df2)}))
        table = reduce(join, frames)

        if method=='pandas':
            table = np.array(table.reorder_levels(columns).to_list(), dtype=itype).T
       
        # If we're using polars and not imploding, collect the lazy dataframe data
        elif not use_implode:
            table = table.select(columns).collect().to_numpy().T

    # [Multiple anchors] Extended dimension tree search over all grids
    # [Multiple non-uniform grids] Extended dimension brute force search optimized with numba
    elif method in ['brute', 'multi', 'tree']:
        if False:#method == 'brute':
            table = brute(
                coordinates,
                resolutions,
                grid_labels,
                axis_labels,
                radius, eps,
                logger,
            )
        else:
        # if True:
            # ftype = max([c.dtype for c in coordinates if np.issubdtype(c.dtype, np.floating)] + [np.float32])

            # Ensure resolutions are in the correct format
            # resolutions = list(map(np.atleast_1d, resolutions))
            if method in ['brute', 'multi']:
                # Format resolutions into (left side, right side) 2D resolution arrays
                for i, res in enumerate(resolutions):
                    c_shp = coordinates[i].shape 

                    # Uniform resolution, simply duplicate for left/right side
                    if res.ndim == 1:
                        res = np.tile(res, (2, 1)).T[None]
                        assert(res.shape == (1, c_shp[-1], 2)), [res.shape, c_shp]
                    
                    # Non-uniform resolution, extend to endpoints for left and right
                    elif res.ndim == 2:
                        assert(res.shape[1] == (c_shp[0]-1)), [res.shape, c_shp]
                        res = np.stack([
                            np.c_[res[:, :1], res].T,
                            np.c_[res, res[:,-1:]].T,
                        ], axis=-1)
                        assert(res.shape[0] == c_shp[0]), [res.shape, c_shp]

                    # Full left/right resolution vectors already provided
                    else: assert(res.shape[0] == c_shp[0]), [res.shape, c_shp]
                    assert(res.shape[-1] == 2), res.shape

                    # Place left/right dimension on the first axis
                    if method != 'multi': 
                        resolutions[i] = np.moveaxis(res, -1, 0)
                    else: resolutions[i] = res

                    # Left/right tolerance is half the resolution (plus a small epsilon)
                    resolutions[i] *= radius
                    resolutions[i] += eps
            else:
                resolutions = [r[..., 0] if len(r.shape) > 2 else r for r in resolutions]
                resolutions = [np.nanmean(r, axis=0) if len(r.shape) > 1 else r for r in resolutions]
                resolutions = list(map(np.atleast_1d, resolutions))

            if method == 'multi':
                table = multiset_single(coordinates, resolutions, num_samples, shuffle, logger)
            else:
                # Optimize column and grid orderings
                optimizations = True
                if optimizations: 
                    make_sizes = lambda c, r: float(f'{r.size}.{c.size}')
                    grid_sizes = starmap(make_sizes, zip(coordinates, resolutions))
                    grid_order = np.argsort(list(grid_sizes))[::-1]
                else: grid_order = list(range(len(coordinates)))
                # grid_order = [2, 3, 5, 0, 4, 1]
                coordinates = [coordinates[i] for i in grid_order]
                resolutions = [resolutions[i] for i in grid_order]
                grid_labels = [grid_labels[i] for i in grid_order]
                axis_labels = [axis_labels[i] for i in grid_order]

                # Create the grids to iterate over and pull out the first query
                grids = zip([c.astype(ftype) for c in coordinates], resolutions)
                query = next(grids)

                # Initialize the table of neighbor indices, along with the column order 
                table = np.arange(len(query[0]), dtype=itype)[:, None]
                order = [0]
                # debug=True
                if debug:
                    print('\nShapes:')
                    with print_table(['Grid', 'Coordinates', 'Resolutions']):
                        for g, c, r in zip(grid_labels, coordinates, resolutions):
                            print('|'.join(map(str, [g, c.shape, r.shape])))

                    for i, (label, values) in enumerate([
                        ('Coordinate', coordinates), 
                        ('Resolution', [r[0] for r in resolutions]),
                    ]):
                        print(f'\n{label} Extents:')
                        with print_table(['Grid', 'Axis', 'Minimum', 'Maximum']):
                            for g, axes, v in zip(grid_labels, axis_labels, values):
                                for a, minim, maxim in zip(axes, v.min(0), v.max(0)):
                                    if (a == 'datetime') and np.isfinite(v).all():
                                        otype = np.timedelta64 if i else np.datetime64
                                        minim = otype(int(minim), 'm')
                                        maxim = otype(int(maxim), 'm')
                                    print('|'.join(map(str, [g, a, minim, maxim])))
                                    g = ''

                    print(f'\nStarting neighbor search with {grid_labels[0]}: {len(table):,} possible matches')

                if logger is not None:
                    logger.debug(f'\tStarting neighbor search with {grid_labels[0]}: {len(table):,} possible matches')

                # Find simultaneously matching indices across all grids
                for i, build in enumerate(grids, 1):
                    if debug:
                        iter_timer = Stopwatch(f'Iteration {i}/{len(coordinates)-1}')
                        iter_timer.__enter__()
                        print(f'\n{iter_timer}')
                        print(''.join(['-']*len(str(iter_timer))))
                        print('Build label:', grid_labels[i])
                        print('Query shape:', query[0].shape, query[1].shape, query[0].dtype)
                        print('Build shape:', build[0].shape, build[1].shape, build[0].dtype)
                        print('Table shape:', table.shape, table.dtype)
                        # print('Virtual dim:', np.isnan(build[0]).all(0).sum(), np.isnan(query[0]).all(0).sum())
                        # print('Unique vals:', list(map(len, map(np.unique, build[0].T))), list(map(len, map(np.unique, query[0].T))))

                    # Duplicate the next grid to align with the current table
                    build_dup = [np.tile(b, i) for b in build]
                    
                    # Extract the build/query coordinates/resolutions
                    b,q,br,qr = chain.from_iterable( zip(*[build_dup, query]) )

                    # Drop any all NaN (virtual) columns from the build and query tables
                    b_nan_col = np.isnan(b).all(0)
                    q_nan_col = np.isnan(q).all(0)
                    skip_dims = b_nan_col | q_nan_col

                    # If requested, swap build/query as necessary to use optimal order
                    if optimizations:
                        # Check virtual dims count (i.e. all values in column are NaN):
                        #  Grid with fewer samples is build when same number of virtual
                        #  Otherwise, grid with more virtual dims is the build
                        b_virt = b_nan_col.sum()
                        q_virt = q_nan_col.sum()
                        switch = (len(b)>len(q)) if q_virt==b_virt else (b_virt<q_virt)
                        if debug: print(f'\t{b_virt=} {q_virt=} {switch=}')

                        # Switch build/query if necessary, based on above criteria
                        bq_order = slice(None, None, -1 if switch else 1)
                    else: bq_order = slice(None, None, 1)
                    (b, br), (q, qr) = [(b, br), (q, qr)][bq_order]

                    # Reorder the columns
                    if optimizations:
                        sort = lambda i,b_col: np.inf if skip_dims[i] else entropy(b_col)[0]
                        cols = np.argsort(list(starmap(sort, enumerate(b.T))))
                        # assert(0), cols
                        b = b[:, cols]
                        q = q[:, cols]
                        skip_dims = skip_dims[cols]

                        # Can't use lexsort until numba's recursion support is fixed
                        # b, b_orig = lexsort(b)
                        # q, q_orig = lexsort(q)
                        # br = (br[:, b_orig] if br.ndim > 1 and br.shape[1] > 1 else br)[..., cols]
                        # qr = (qr[:, q_orig] if qr.ndim > 1 and qr.shape[1] > 1 else qr)[..., cols]

                        b_orig = np.lexsort(b.T[::-1])
                        q_orig = np.lexsort(q.T[::-1])
                        # b_orig = np.arange(len(b))[bo]
                        # q_orig = np.arange(len(q))[qo]
                        b = b[b_orig]
                        q = q[q_orig]
                        br = (br[:, b_orig] if br.ndim > 1 and br.shape[1] > 1 else br)[..., cols]
                        qr = (qr[:, q_orig] if qr.ndim > 1 and qr.shape[1] > 1 else qr)[..., cols]

                        if debug:
                            print(f'After optimizations:')
                            print('\tQuery shape:', q.shape, qr.shape, q.dtype)
                            print('\tBuild shape:', b.shape, br.shape, b.dtype)
                            # print('\tVirtual dim:', np.isnan(b).all(0).sum(), np.isnan(q).all(0).sum())
                            # print('\tUnique vals:', list(map(len, map(np.unique, b.T))), list(map(len, map(np.unique, q.T))))
                    
                    if debug:
                        print('  b q shape:', b.shape, q.shape)
                        print('br qr shape:', br.shape, qr.shape)
                        print('  skip dims:', skip_dims)

                    # Double check that build and query are already lexographically sorted
                    # assert((q[np.lexsort(q.T[::-1])] == q).all()), [q, q[np.lexsort(q.T[::-1])]]
                    # assert((b[np.lexsort(b.T[::-1])] == b).all()), [b, b[np.lexsort(b.T[::-1])]]

                    # Find the matching indices between the build/query grids
                    if method == 'brute':
                        progress = nullcontext(None)
                        if debug or (logger is not None):
                            if ProgressBar is not None:
                                streamer = None if debug else StreamToLogger(logger, logging.DEBUG)
                                progress = ProgressBar(
                                    total=len(b), 
                                    file=streamer, 
                                    update_interval=10, 
                                    dynamic_ncols=False,
                                    notebook=False, 
                                    postfix='find_neighbors progress',
                                )
                        # progress = ProgressBar(total=len(b), update_interval=1)

                        with progress as pbar:
                            # b_ix, q_ix = bruteforce_double(b, q, *br, *qr, skip_dims, pbar)[bq_order]
                            b_ix, q_ix = bruteforce_single(b, q, *br, *qr, skip_dims, pbar)[bq_order] # Original w/ dict
                            # b_ix, q_ix = bruteforce_original(b, q, *br, *qr, skip_dims, pbar).T[bq_order] # Original
                        if debug: print('b q i shape:', b_ix.shape, q_ix.shape)
                    else:
                        b,q,br,qr = [arr[..., ~skip_dims] for arr in [b,q,br,qr]]
                        b_ix,q_ix = get_matches([b,q], [br,qr], expand=True)[bq_order]

                    # If no matches are found, we can immediately return 
                    if min(b_ix.size, q_ix.size) == 0: 
                        if logger: logger.debug(f'\tNo matches left after {grid_labels[i]}')
                        if debug:  print(f'\tNo matches left after {grid_labels[i]}')
                        return (np.empty((0, 0)),) * 2

                    if optimizations:
                        b_orig, q_orig = [b_orig, q_orig][bq_order]
                        b_ix = b_orig[b_ix]
                        q_ix = q_orig[q_ix]

                    # Save time/memory by skipping unnecessary work on the final loop
                    if (i+1) < len(coordinates):
                        _,bqr = (b, q), (br, qr) = list(zip(build, query)) 

                        # If resolution can be non-uniform (i.e. brute method)
                        if max(br.ndim, qr.ndim) > 1:

                            # If either grid uses a non-uniform resolution, both need to
                            if max(br.shape[1], qr.shape[1]) > 1:
                                if br.shape[1] == 1: br = np.tile(br, (1, len(b), 1))
                                if qr.shape[1] == 1: qr = np.tile(qr, (1, len(q), 1))
                                br_qr = [br[:, b_ix], qr[:, q_ix]]
                            else: br_qr = bqr
                            br_qr = np.dstack(br_qr[bq_order])
                        else: br_qr = np.hstack(bqr[bq_order])

                        # Construct the next query set by combining the current two grids
                        query = np.c_[(b[b_ix], q[q_ix])[bq_order]], br_qr
                        if debug: print(' next query:', query[0].shape, query[1].shape)

                    # Separate the coordinates/resolutions and extract the locations
                    table = np.c_[(b_ix, table[q_ix])[bq_order]]
                    order = sum([[i], order][bq_order], [])

                    if debug: 
                        print(' next table:', table.shape)
                        iter_timer.__exit__()
                    if logger is not None:
                        logger.debug(f'\t{len(table):>11,} matches remain after {grid_labels[i]}')

                # Reorder the columns correctly 
                if debug: print('\nReordering table...')
                table = table[:, np.argsort(order)[np.argsort(grid_order)]]

        if table.size == 0:
            return (np.empty((0, 0)),) * 2

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
        count = None#np.ones_like(table, dtype=itype)
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
        
