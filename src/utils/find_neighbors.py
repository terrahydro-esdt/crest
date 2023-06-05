from sklearn.neighbors import BallTree
from collections.abc import Collection
from itertools import combinations
from functools import reduce

import pandas as pd
import numpy as np


def find_neighbors(
    coordinates : Collection[np.ndarray], 
    resolutions : Collection[np.ndarray] | None = None,
    radius      : float = 0.5,
    allow_empty : bool  = False,
    **kwargs,
) -> (np.ndarray, np.ndarray):
    """ Find all nearest neighbors for the given coordinates.

    Notes
    -----
    Ordinarily, finding nearest neighbors for multiple sets of coordinate 
    grids would have combinatorial time complexity. By leveraging BallTrees
    however, we can reduce that to K*Dlog(N), where K is the number of grids,
    D is the dimensionality of the grids, and N is the size of the grids. 

    To find nearest neighbors, this method builds N-1 BallTrees and queries
    each of them using a reference coordinate set (where N=len(coordinates)).
    The reference coordinate set is the first index in the given collection
    of coordinates.

    Parameters
    ----------
    coordinates : Collection[np.ndarray]
        Coordinate locations to find neighbors for.
    resolutions : Collection[np.ndarray] | None
        Resolution of the given coordinate locations, which is used
        to normalize the coordinates prior to building the BallTrees.
        If it is not given, no normalization is used.
    radius      : float
        Radius in which points are considered neighbors of a reference point.
    allow_empty : bool
        Whether to allow empty neighbor sets in the results (i.e. all reference
        points are returned, regardless of if there are any neighbors). By
        default, only reference points which have at least one neighbor in 
        all other coordinate sets are returned.
    **kwargs
        Additional keywords are passed to sklearn.neighbors.BallTree.

    Returns
    -------
    (np.ndarray, np.ndarray)
        A tuple of two arrays: neighbor indices, and neighbor counts.
        Both arrays are shaped (len(coordinates), len(coordinates[0])),
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
    def mask(c1, c2):
        """ Filter out axes which contain all nan values """
        def nan_axis(c):
            """ If an axis contains nan values, all axis values must be nan """
            nans = ~np.isfinite(c)
            assert(np.where(nans.any(0), nans.all(0), True).all())
            return np.where(nans.any(0))[0].tolist()
        invalid = nan_axis(c1) + nan_axis(c2)
        return c1[:, tuple(i for i in range(c1.shape[1]) if i not in invalid)] 


    # def get_matches(idxs):
    #     rs = [resolutions[i] for i in idxs]
    #     cs = [coordinates[i] for i in idxs]

    #     shape = lambda c: c.reshape(len(c), -1)
    #     scale = lambda c: c / np.nanmax(rs, axis=0, keepdims=True)

    #     anchor, query = map(mask, map(scale, map(shape, cs)), cs[::-1])

    #     match = BallTree(anchor, **kwargs).query_radius(query, radius + 1e-5)
    #     frame = dict(zip(np.arange(len(query)), match))
    #     frame = pd.DataFrame.from_dict(frame, orient='index').stack()
    #     frame = frame.reset_index().set_index(['level_0', 0])
    #     frame.index   = frame.index.set_names(idxs[::-1])
    #     frame.columns = [str(idxs)]
    #     return frame
        # row_1 = list( chain.from_iterable(match) )
        # row_2 = np.repeat(np.arange(len(query)), list(map(len, match)))
        # match = -np.ones((len(coordinates), len(row_1)), dtype='int32')
        # match[list(idxs)] = [row_1, row_2]
        # return idxs, match

    
    # def check_match(matches, match):
    #     seen, matches = matches
    #     (a, b), match = match 
    #     assert((a in seen) or (b in seen))
        
    #     # Create the inner join mask where we already have matches
    #     mask = (a not in seen) or matches[a][:, None] == match[a]
    #     mask&= (b not in seen) or matches[b][:, None] == match[b]
    #     seen = set(seen) | {a, b}

    #     # Combine old and new match indices for current rows
    #     m_1,m_2 = np.where(mask)
    #     matches = matches[:, m_1]
    #     matches[[a,b]] = match[[a,b]][:, m_2]
    #     return seen, matches

    # Perform an inner join on matches over all combinations of grids
    # resolutions = resolutions or [1] * len(coordinates)
    # paired_rows = combinations(range(len(coordinates)), 2)
    # _,  matches = reduce(check_match, map(get_matches, paired_rows))


    # Perform an inner join on matches over all combinations of grids
    # resolutions = resolutions or [1] * len(coordinates)
    # paired_rows = list(combinations(range(len(coordinates)), 2))

    # print(f'getting matches for {len(paired_rows)} pairs')
    # matches = list(map(get_matches, paired_rows))
    # print(f'got {len(matches)} matches: {[m.shape for m in matches]}')
    # pd_join = lambda df1, df2: df1.join(df2, how='inner')
    # matches = reduce(pd_join, matches).reorder_levels(range(len(coordinates)))

    # return [np.array(m)[:,None] for m in matches.index.to_numpy()]

    """
    Below implementation returns an outer product over the grid combinations,
    using the reference grid as the anchor; above implementation returns the 
    inner product over grid combinations, using every grid as an anchor at 
    least once.
    """
    # Reference set is the first element
    res, *resolutions = resolutions or [1] * len(coordinates)
    ref, *coordinates = coordinates

    # Add small value to radius to account for numerical instability
    # Bear in mind this is related to the resolution's rounding
    radius += 1e-5

    # Create BallTrees and query against the reference locations
    #    norm : Normalize by the max resolution (max[current, reference])
    #   build : Construct the BallTree using normalized coordinates
    #   query : Query for neighbors within `radius` of the reference
    #   match : Contains neighbors of reference for each set of coordinates
    scale = lambda c, r: c / np.nanmax([res, r], axis=0, keepdims=True)
    build = lambda c, r: BallTree(mask(scale(c,r), ref), **kwargs).query_radius
    query = lambda c, r: build(c, r)(mask(scale(ref, r), c), radius)
    match = list(map(query, coordinates, resolutions))

    # Include the reference set indices in the final list of neighbors
    r_ix    = np.empty(len(ref), dtype=object)
    r_ix[:] = list(np.arange(len(ref), dtype='int64').reshape(-1, 1))
    indices = np.c_[[r_ix] + match]
    counts  = np.array([[1] * len(ref)] + [list(map(len, m)) for m in match])

    # Filter neighbor lists in which any of the grids are missing
    if (len(indices) > 1) and (not allow_empty):
        mask    = np.all(counts[1:] > 0, 0)
        counts  = counts[:, mask]
        indices = indices[:, mask]

    # Both indices and counts are shaped (num coordinate grids, len(ref grid))
    # Indices is a ragged object array however, where each element is a
    # variable length array:
    # [ 
    #    array(ref grid indices) 
    #    array([grid 2 indices matching ref grid indices[0]], 
    #          [grid 2 indices matching ref grid indices[1]],
    #          [...])
    #    array([grid 3 indices matching ref grid indices[0]],
    #          [grid 3 indices matching ref grid indices[1]],
    #          [...])
    # ]
    # For example, indices equal to 
    #     [ [[1], [2]],  [[0, 1], [2]],  [[2], [4,5]] ]
    # would indicate:
    #  - ref_grid[1] matches [grid_2[0], grid_2[1]] and [grid_3[2]]
    #  - ref_grid[2] matches [grid_2[2]] and [grid_3[4], grid_3[5]]
    #
    # Counts is then the length (number of matches) for each grid. Using
    # the above example, counts would correspond to:
    #     [[1, 1], [2, 1], [1, 2]]
    # As the matches for the ref grid have lengths 1 and 1; for grid_2 
    # have lengths 2 and 1; and for grid_3 have lengths 1 and 2. 
    return indices, counts

