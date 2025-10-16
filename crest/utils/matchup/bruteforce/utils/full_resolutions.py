import numpy as np


def full_resolutions(
    coordinates: np.ndarray, 
    resolutions: np.ndarray,
) -> np.ndarray:
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
    n_coord = len(coordinates)
    n_dims  = coordinates.shape[-1]
    r_shape = resolutions.shape 

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
    assert(full.shape[1] in [1, n_coord]), [full.shape, r_shape, n_coord]
    assert(full.shape[2] == n_dims),       [full.shape, r_shape, n_dims]
    return full