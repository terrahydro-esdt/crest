import numba as nb 
import numpy as np 


@nb.njit([
    'UniTuple(int32, 2)(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])',
    'UniTuple(int32, 2)(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])',
], cache=True)
def check(x, xrl, xrr, y, yrl, yrr):
    """ lambda a,b,c,x,y,z: ((b+a)/2 <= y <= (c+b)/2) or ((y+x)/2 <= b <= (z+y)/2) 
        
        Returns
        -------
        int, int
            - Dimension which caused the check to fail (-1 if the check succeeds)
            - Flag specifying if the failure is due to being above (1) or below (0)

    """
    for i in range(len(x)):
        if (y[i]-yrl[i]) > x[i]:
            if (x[i]-xrl[i]) > y[i]:
                return i, 0
            if y[i] > (x[i]+xrr[i]):
                return i, 1

        elif x[i] > (y[i]+yrr[i]):
            if (x[i]-xrl[i]) > y[i]:
                return i, 0
            if y[i] > (x[i]+xrr[i]):
                return i, 1
    return -1, -1



@nb.njit([
    'int32[:,:](float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])',
    'int32[:,:](float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])',
], cache=True, parallel=True, nogil=True)
def bruteforce(a1, a2, a1l, a1r, a2l, a2r):
    """ Brute-force method to find neighbors within left/right distance tolerance.
    
    Note that this method *requires* a1 and a2 to be lexicographically sorted,
    or else some valid matches may be missed. For our application, this 
    constraint can be adhered to for free - so long as care is taken when 
    combining grids together into their extended-dimension representation;
    e.g. bruteforce(A, B) -> extended-dimension grid AB.
         bruteforce(AB, C) -> ABC
         etc.

    The remaining l/r parameters correspond to the left and right side neighbor
    distance tolerance for a1 and a2, respectively. 

    """
    l1,l2 = len(a1), len(a2)
    skips = np.zeros_like(a2, dtype=np.int32)
    match = [(np.int32(x), np.int32(x)) for x in range(0)]

    # Create skip list, which gives index of next unique value
    for dim in nb.prange(a2.shape[1]): 
        value = a2[-1, dim]
        index = l2

        for i in range(l2-1, -1, -1):
            if a2[i, dim] != value:
                value = a2[i, dim]
                index = i+1
            skips[i, dim] = index
    
    # Take the minimum to the left for each element
    # skips = np.minimum.accumulate(skips, axis=-1)
    for row in nb.prange(l2):
        minim = np.inf 
        for i in range(skips.shape[1]):
            skips[row, i] = minim = min(minim, skips[row, i])

    # Allow static resolution to be used
    dynamic_r1 = len(a1l) == l1
    dynamic_r2 = len(a2l) == l2
    if not dynamic_r1: 
        a1ls = a1l[0]
        a1rs = a1r[0]
    if not dynamic_r2:
        a2ls = a2l[0]
        a2rs = a2r[0]

    # Iterate all elements in a1
    start = 0 
    for i in range(l1):
        x   = a1[i]
        xrr = a1r[i] if dynamic_r1 else a1rs
        xrl = a1l[i] if dynamic_r1 else a1ls
        new = False

        # Iterate all elements in a2, skipping any from the prior a1 step
        j = start
        while j < l2:
            y   = a2[j]
            yrr = a2r[j] if dynamic_r2 else a2rs
            yrl = a2l[j] if dynamic_r2 else a2ls

            invalid_dim, above = check(x, xrl, xrr, y, yrl, yrr)

            # This is only skipping based on the first dimension, but
            # implementing the full dimension skipping logic has a lot of
            # edge cases to handle. For now, the additional speed isn't
            # worth the time required. That may change if much larger number
            # of dimensions needs to be handled (as allowing logic to skip
            # dims other than the first would net larger improvements)
            if (invalid_dim != 0) and not new:
                start = j
                new   = True

            # If this isn't a match, skip forward
            if invalid_dim > -1:

                # If the current a2 invalid dimension value is above a1's
                # value, skip forward based on the prior dimension's skip list
                if above:
                    invalid_dim -= 1

                    # If we can no longer skip forward, move on to the next
                    # a1 value and skip back to the first candidate found
                    # during this round of a2 iteration
                    if invalid_dim < 0: break

                # Skip forward based on whichever dimension caused the failure
                # (or the dimension prior to it, if we're above all candidates)
                j = skips[j][invalid_dim]

            # Store match indices and increment the index
            else: 
                match.append((i,j))
                j = j + 1 

    if len(match) == 0:
        return np.empty((0,2), dtype=np.int32)
    return np.array(match, dtype=np.int32)


@nb.njit(cache=True, parallel=True)
def interleave(a, b):
    """ Interleave the last dimensions of a and b:
      Rather than np.c_[a=[1,3,5], b=[2,4,6]] -> [1,3,5,2,4,6]
      we instead create -> [1,2,3,4,5,6]
    """
    a_s, a_f = a.shape[:-1], a.shape[-1]
    b_s, b_f = b.shape[:-1], b.shape[-1]

    # All dims except the last must be equal
    assert(a_s == b_s), [a.shape, b.shape]

    # Check that the number of features are compatible with interleaving
    assert(((a_f % b_f) == 0) or ((b_f % a_f) == 0)), f'{a_f} and {b_f} are not interleavable'

    c = np.zeros(a.shape[:-1] + (a.shape[-1] + b.shape[-1],), dtype=a.dtype)
    x = max(1, a_f//b_f)
    y = max(1, b_f//a_f)

    for i in nb.prange(x): c[...,   i::x+y] = a[..., i::x]
    for j in nb.prange(y): c[..., x+j::x+y] = b[..., j::y]
    return c 