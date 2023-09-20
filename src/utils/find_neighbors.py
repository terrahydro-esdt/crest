from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from collections.abc import Collection
from itertools import combinations, product, chain
from functools import reduce, partial

import numpy as np
import polars as pl
import pandas as pd 

from .Stopwatch import Stopwatch 
from ._bruteforce import bruteforce
import time


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


def implode(table: np.ndarray | pl.DataFrame):
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
    # Exclude one name from the full list
    # Groupby all names except one, then concat into a comma-delimited string
    excl = lambda remove: list( set(names) - {remove} )
    join = lambda tbl, n: tbl.groupby(excl(n)).agg(pl.col(n).str.concat(','))

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
    return ( reduce(join, names[::-1], table)
        .sort(names)
        .with_columns(
            pl.col(names).str.split(',')
              .cast(pl.List(dtype)))
        .select(names)
        .collect().to_numpy().T # Ensure pyarrow installed if error
    )


def find_neighbors(
    coordinates : Collection[np.ndarray], 
    resolutions : Collection[np.ndarray] | None = None,
    radius      : float = 0.5,
    method      : str   = 'brute',
    allow_empty : bool  = False,
    use_implode : bool  = False,
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
    of the grids, and N is the size of the grids.

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
        Radius in which points are considered neighbors of a reference point.
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
            could be made to the method as well, if warranted. 
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

    **kwargs
        Additional keywords are passed to sklearn.neighbors.BallTree.

    Returns
    -------
    (np.ndarray, np.ndarray)
        A tuple of two arrays: neighbor indices, and neighbor counts.
        Both arrays are shaped [len(coordinates), len(coordinates[0])],
        but neighbor indices (first array) is a ragged object array where 
        each element is itself a variable length array:
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
        correspond to:
            [[1, 1], [2, 1], [1, 2]]
        as the matches for the ref grid have lengths 1 and 1; for grid_2 
        have lengths 2 and 1; and for grid_3 have lengths 1 and 2. 

    """

    # Rough guess on what dtype can be used to hold indices
    large = sum(map(np.log10, map(len, coordinates))) > 17
    dtype = np.int64 if large else np.int32

    # Add small value to radius to account for numerical instability
    # Bear in mind this is related to the resolution's rounding
    kwargs.update({
        'radius' : radius + 1e-5,
        'dtype'  : dtype,
    })

    # Set the radius, dtype, and any other kwargs given
    get_matches = partial(get_indices, **kwargs)

    # Set a default value for the resolutions if None was given
    resolutions = resolutions or [np.ones(c.shape[-1]) for c in coordinates]

    if len(coordinates) == 1:
        table = np.arange(len(coordinates[0]), dtype=dtype)[None]
        count = np.ones_like(table)
        return table, count 

    # Single anchor grid, checked against all other grids
    if method == 'anchor':
        # Create trees and query against the reference (anchor) grid
        coordinates = product(coordinates[1:], coordinates[:1])
        resolutions = product(resolutions[1:], resolutions[:1])
        match = list(map(get_matches, coordinates, resolutions))
        n_ref = len(match[0])

        # Include the reference set indices in the final list of neighbors
        ix    = np.empty(n_ref, dtype=object)
        ix[:] = list(np.arange(n_ref, dtype=dtype)[:, None])
        table = np.c_[[ix] + match]
        count = np.array([[1]*n_ref] + [list(map(len, m)) for m in match], dtype=dtype)

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
            if use_polars:
                return pl.DataFrame(dict(zip(keys, [build_i, query_i]))).set_sorted(keys[1]).lazy()
            return pd.MultiIndex.from_arrays([build_i, query_i], names=keys)

        # Perform an inner join on matches over all combinations of grids
        indices = list(range(len(coordinates)))
        columns = [f'col_{i}' for i in indices]
        pairing = partial(combinations, r=2)
        frames  = map(build_frame, *map(pairing, [columns, coordinates, resolutions]))

        c_set = lambda df1, df2: list( set(df2.columns) & set(df1.columns) )
        join  = lambda df1, df2: df1.join(df2, how='inner', **({} if use_pandas else {'on': c_set(df1, df2)}))
        table = reduce(join, frames)

        if use_pandas:
            table = np.array(table.reorder_levels(columns).to_list(), dtype=dtype).T
       
        # If we're using polars and not imploding, collect the lazy dataframe data
        elif not use_implode:
            table = table.select(columns).collect().to_numpy().T

    # [Multiple anchors] Extended dimension tree search over all grids
    elif method == 'tree':
        grids = zip(coordinates, map(np.atleast_1d, resolutions))
        query = next(grids)

        # Initialize the final array of neighbor indices
        table = np.arange(len(query[0]), dtype=dtype)[:, None]

        # Find simultaneously matching indices across all grids
        for i, build in enumerate(grids, 1):
            order = slice(None, None, 1 if len(build[0]) > len(query[0]) else -1)
            tiled = [np.tile(b, i) for b in build]
            cs_rs = zip(*[tiled, query][order])
            
            # Update the table indices to include the new grid
            build_ix, query_ix = get_matches(*cs_rs, expand=True)[order]
            (build, query), rs = zip(build, query) 

            # Collate the new query values, resolutions, and indices
            table = np.c_[table[query_ix], build_ix]
            query = np.c_[query[query_ix], build[build_ix]], np.hstack(rs)

        table = table.T

    # [Multiple non-uniform grids] Extended dimension brute force search optimized with numba
    elif method == 'brute':
        ftype = max([c.dtype for c in coordinates if np.issubdtype(c.dtype, np.floating)] + [np.float32])

        # Format resolutions into (left side, right side) 2D resolution arrays
        resolutions = list(map(np.atleast_1d, resolutions))
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
            res = np.moveaxis(res, -1, 0)

            # Left/right tolerance is half the resolution (plus a small epsilon)
            resolutions[i] = res.astype(ftype) * radius + 1e-5

        # Create the grids to iterate over and pull out the first query
        grids = zip([c.astype(ftype) for c in coordinates], resolutions)
        query = next(grids)

        # Initialize the table of neighbor indices, along with the column order 
        table = np.arange(len(query[0]), dtype=np.int32)[:, None]
        order = [0]

        # Find simultaneously matching indices across all grids
        for i, build in enumerate(grids, 1):

            # Duplicate the next grid to align with the current table
            build_dup = [np.tile(b, i) for b in build]

            # Order grids so that the one with fewer elements is first (used as the build)
            min_first = slice(None, None, 1 if len(build[0]) < len(query[0]) else -1) 

            # Extract the build/query coordinates/resolutions
            b,q,br,qr = chain.from_iterable( zip(*[build_dup, query][min_first]) )
            
            # Find the matching indices between the build/query grids
            # assert((q[np.lexsort(q.T[::-1])] == q).all())#, [q, q[np.lexsort(q.T[::-1])]]
            # assert((b[np.lexsort(b.T[::-1])] == b).all())#, [b, b[np.lexsort(b.T[::-1])]]
            b_ix, q_ix = bruteforce(b, q, *br, *qr).T[min_first]

            # Separate the coordinates/resolutions and extract the locations
            table = np.c_[(b_ix, table[q_ix])[min_first]]
            order = sum([[i], order][min_first], [])

            # Save time/memory by skipping unnecessary work on the final loop
            if (i+1) < len(coordinates):
                _,bqr = (b, q), (br, qr) = list(zip(build, query)) 

                # If either grid uses a non-uniform resolution, both need to
                if max(br.shape[1], qr.shape[1]) > 1:
                    if br.shape[1] == 1: br = np.tile(br, (1, len(b), 1))
                    if qr.shape[1] == 1: qr = np.tile(qr, (1, len(q), 1))
                    br_qr = [br[:, b_ix], qr[:, q_ix]]
                else: br_qr = bqr

                # Construct the next query set by combining the current two grids
                query = np.c_[(b[b_ix], q[q_ix])[min_first]], np.dstack(br_qr[min_first])

        # Reorder the columns correctly 
        table = table[:, np.argsort(order)].T
       
    if use_implode:
        table = implode(table)
        count = np.array([list(map(len, col)) for col in table], dtype=dtype)
    else:
        count = np.ones_like(table, dtype=dtype)

    return table, count