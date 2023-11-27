from numba import (
    int32   as i32,
    int64   as i64,
    float32 as f32,
    float64 as f64,
    boolean,
    optional,
)
import numba as nb 
import numpy as np 

# Allow bruteforce progress logs if numba_progress is installed
try:
    from numba_progress.progress import ProgressBarType
    pb_type = optional(ProgressBarType)
    @nb.njit((pb_type, i64), cache=True)
    def update(pbar, i): pbar.set(i)
except ImportError:
    pb_type = optional(i64)
    @nb.njit((pb_type, i64), cache=True)
    def update(pbar, i): pass



@nb.njit([
    nb.types.UniTuple(i32, 2)(fx[:], fx[:], fx[:], fx[:], fx[:], fx[:])
    for fx in [f32, f64]
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
    i32[:,:](fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], boolean[:], pb_type)
    for fx in [f32, f64]
], cache=True, parallel=True, nogil=True)
def bruteforce(a1, a2, a1l, a1r, a2l, a2r, skipdim, progress=None):
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
    # skip1 = np.zeros_like(a1, dtype=np.int32)
    skip2 = np.zeros_like(a2, dtype=np.int32)
    match = [(np.int32(x), np.int32(x)) for x in range(0)]

    # Create skip list, which gives index of next unique value
    # for dim in nb.prange(a1.shape[1]): 
    #     value = a1[-1, dim]
    #     index = l1

    #     for i in range(l1-1, -1, -1):
    #         if not (np.isnan(value) and np.isnan(a1[i,dim])) and (a1[i, dim] != value):
    #             value = a1[i, dim]
    #             index = i+1
    #         skip1[i, dim] = index

    for dim in nb.prange(a2.shape[1]): 
        value = a2[-1, dim]
        index = l2

        for i in range(l2-1, -1, -1):
            if not (np.isnan(value) and np.isnan(a2[i,dim])) and (a2[i, dim] != value):
                value = a2[i, dim]
                index = i+1
            skip2[i, dim] = index

    # for dim in range(a2.shape[1]):
    #     print('skip1', dim, len(np.unique(skips[:,dim])))
    # Take the minimum to the left for each element
    # skip2 = np.minimum.accumulate(skip2, axis=-1)
    # for row in nb.prange(l1):
    #     minim = np.inf 
    #     for i in range(skip1.shape[1]):
    #         skip1[row, i] = minim = min(minim, skip1[row, i])

    for row in nb.prange(l2):
        minim = np.inf 
        for i in range(skip2.shape[1]):
            skip2[row, i] = minim = min(minim, skip2[row, i])

    a1 = a1[..., ~skipdim]
    a2 = a2[..., ~skipdim]
    a1l = a1l[..., ~skipdim]
    a2l = a2l[..., ~skipdim]
    a1r = a1r[..., ~skipdim]
    a2r = a2r[..., ~skipdim]

    indices = np.append(np.cumsum(skipdim)[~skipdim], 0)

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
    i = start = 0 
    while i < l1:
        x   = a1[i]
        xrr = a1r[i] if dynamic_r1 else a1rs
        xrl = a1l[i] if dynamic_r1 else a1ls
        new = False
        # lst = -1 
        # starts = [-1] * len(skipdim)

        # if mark > -1: print('\nStarting at j =', start)

        # Iterate all elements in a2, skipping any from the prior a1 step
        j = start
        while j < l2:
            # print()
            # input(f'Continue i={i}  j={j} ?')

            # if progress is not None: 
            #     # update(progress, )
            #     progress.update(1)

            y   = a2[j]
            yrr = a2r[j] if dynamic_r2 else a2rs
            yrl = a2l[j] if dynamic_r2 else a2ls

            invalid_dim, above = check(x, xrl, xrr, y, yrl, yrr)
            invalid_dim += indices[invalid_dim]
            # print(f'{x}  vs  {y}')
            # print(f'invalid dim: {invalid_dim}  y above x: {above == 1}')

            # This is only skipping based on the first dimension, but
            # implementing the full dimension skipping logic has a lot of
            # edge cases to handle. For now, the additional speed isn't
            # worth the time required. That may change if much larger number
            # of dimensions needs to be handled (as allowing logic to skip
            # dims other than the first would net larger improvements)
            # if (j == k) and (above == 1):
            # #     # if i > 8000: print(f'\n({i}, {j}):  {x}  vs  {y}')
            #     i = skip1[i][invalid_dim] - 1
            #     # if i > 8000: 
            #     #     print(f'invalid dim: {invalid_dim}  y above x: {above == 1}')
            #     #     print(f'breaking; continuing at i={i+1}  j={start}')
            #     #     print(f'current matches: {len(match)}')
            #     # break 

            if (invalid_dim != 0) and not new:
                new   = True
                start = j
                # print(f'set start = {start}')

            # If this isn't a match, skip forward
            if invalid_dim > -1:

                # If the current a2 invalid dimension value is above a1's
                # value, skip forward based on the prior dimension's skip list
                if above:
                    invalid_dim -= 1

                    # If we can no longer skip forward, move on to the next
                    # a1 value and skip back to the first candidate found
                    # during this round of a2 iteration
                    if invalid_dim < 0: 
                        # print(f'\n({i}, {j}):  {x}  vs  {y}')
                        # j = start
                        # y   = a2[j]
                        # yrr = a2r[j] if dynamic_r2 else a2rs
                        # yrl = a2l[j] if dynamic_r2 else a2ls

                        # invalid_dim, above = check(x, xrl, xrr, y, yrl, yrr)
                        # invalid_dim += indices[invalid_dim]
                        # if above == 1:
                        #     i = skip1[i][invalid_dim] - 1
                        # elif above == 0:
                            # start = skip2[start][invalid_dim]
                        # print(f'invalid dim: {invalid_dim}  y above x: {above == 1}')
                        # print(f'breaking; continuing at i={i+1}  j={start}')
                        # print(f'current matches: {len(match)}')
                        break

                # Skip forward based on whichever dimension caused the failure
                # (or the dimension prior to it, if we're above all candidates)
                j = skip2[j][invalid_dim]
                # print(f'setting j = {j}')

            # Store match indices and increment the index
            else: 
                match.append((i,j))
                j = j + 1 
        i = i + 1

        # if i > 14: 
        #     print(match)
        #     assert(0)
        if progress is not None: 
            update(progress, i)
    # print(f'matches: {len(match)}\n\n')
    # # if len(match) != 10752: assert(0)
    # assert(0)

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