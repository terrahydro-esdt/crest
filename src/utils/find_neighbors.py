from sklearn.neighbors import BallTree
from collections.abc import Collection

import numpy as np


def find_neighbors(
    coordinates : Collection[np.ndarray], 
    resolutions : Collection[np.ndarray] | None = None,
    radius      : float = 0.5,
    allow_empty : bool  = False,
    **kwargs,
) -> list[tuple[np.ndarray]]:
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
    list[tuple[np.ndarray]]
        Neighbor indices for each element of the reference, including itself.
        There will be one tuple per reference index in the output list if
        allow_empty=True, with any tuples containing empty arrays (i.e. no
        neighbors for that point) being filtered if allow_empty=False. Tuples
        contain the reference index at tuple index 0, and one numpy array 
        for every other coordinate set passed. These remaining numpy arrays
        contain the indices into the respective coordinate set which match
        the reference at the given reference index. For example: 
        [([3], [1], [2,4])] would indicate reference index 3 matches index
        1 in the second coordinate set, and indices 2 and 4 in the third 
        coordinate set.  

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
    neighbors = [np.arange(len(ref)).reshape(-1, 1)] + match
    return [n for n in zip(*neighbors) if allow_empty or all(map(len, n))]
