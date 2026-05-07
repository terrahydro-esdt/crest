from numba import (
    int32   as i32,
    int64   as i64,
    float32 as f32,
    float64 as f64,
    boolean,
    optional,
)
from numba.types import UniTuple, Tuple
from numba.typed.typeddict import DictType, Dict
from numba.typed.typedlist import ListType, List

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

INPUT_TYPES = [f32, f64, i32, i64][:2]


@nb.njit([
    UniTuple(i32, 2)(fx[:], fx[:], fx[:], fx[:], fx[:], fx[:], boolean)
    for fx in INPUT_TYPES
], cache=True, nogil=True)
def check(x, xrl, xrr, y, yrl, yrr, use_locus):
    """ lambda a,b,c,x,y,z: ((b+a)/2 <= y <= (c+b)/2) or ((y+x)/2 <= b <= (z+y)/2) 
        
        Returns
        -------
        int, int
            - Dimension which caused the check to fail (-1 if the check succeeds)
            - Flag specifying if the failure is due to being above (1) or below (0)

    """
    # count = 0
    for i in range(len(x)):
        if use_locus:
            if (x[i]-xrl[i]) > (y[i]+yrr[i]):
                return i, 0
            if (y[i]-yrl[i]) > (x[i]+xrr[i]):
                return i, 1
        else:
            # count += 1
            if (y[i]-yrl[i]) > x[i]:
                # count += 1
                if (x[i]-xrl[i]) > y[i]:
                    return i, 0#, count
                # count += 1
                if y[i] > (x[i]+xrr[i]):
                    return i, 1#, count
            else:
                # count += 1
                if x[i] > (y[i]+yrr[i]):
                    # count += 1
                    if (x[i]-xrl[i]) > y[i]:
                        return i, 0#, count
                    # count += 1
                    if y[i] > (x[i]+xrr[i]):
                        return i, 1#, count

    return -1, -1#, count


@nb.njit(cache=True, parallel=True)
def interleave(a, b):
    """ Interleave the last dimensions of a and b.
    
    Rather than np.c_[a=[1,3,5], b=[2,4,6]] -> [1,3,5,2,4,6]
    we instead create -> [1,2,3,4,5,6]
    
    """
    a_s, a_f = a.shape[:-1], a.shape[-1]
    b_s, b_f = b.shape[:-1], b.shape[-1]

    # All dims except the last must be equal
    assert(a_s == b_s), [a.shape, b.shape]

    # Check that the number of features are compatible with interleaving
    assert(((a_f % b_f) == 0) or ((b_f % a_f) == 0)), (
        f'{a_f} and {b_f} are not interleavable')

    c = np.zeros(a.shape[:-1] + (a.shape[-1] + b.shape[-1],), dtype=a.dtype)
    x = max(1, a_f//b_f)
    y = max(1, b_f//a_f)

    for i in nb.prange(x): c[...,   i::x+y] = a[..., i::x]
    for j in nb.prange(y): c[..., x+j::x+y] = b[..., j::y]
    return c 


@nb.njit([i32[:, :](fx[:, :], fx[:, :], fx[:, :]) for fx in INPUT_TYPES], 
    cache=True, nogil=True)
def create_skiplist(array, res_l, res_r):
    """ Create a skip list for the given array.

    The skip list indicates the index for the next unique
    value along each axis. For example, the array::

        [ [1 1 2 0]
          [1 2 2 2]
          [2 1 1 2] ]

    would generate the skip list::

        [ [2 1 2 1]
          [2 2 2 3]
          [3 3 3 3] ]

    However, we then also set each element to the minimum
    value left of the of the element in its row,
    i.e. `skip = np.minimum.accumulate(skip, axis=-1)`::

        [ [2 1 1 1]
          [2 2 2 2]
          [3 3 3 3] ]

    This ensures no potential matches are missed when the
    skip list is used to jump forward during bruteforce.

    """

    # Include extra last column for skipping on column=-1
    x, y = array.shape
    skip = np.zeros((x, y+1), dtype=np.int32)
    skip[:, y] = x

    dynl = len(res_l) == x
    dynr = len(res_r) == x

    # First construct the raw skip list values
    for col in nb.prange(y): 
        value = array[-1, col]
        index = skip[x-1, col] = x

        for row in range(x-2, -1, -1):
            value2 = array[row, col]

            if not (np.isnan(value) and np.isnan(value2)):
                if (value2 != value):
                    value = value2
                    index = row + 1

                # If resolution varies, ensure right resolution is handled:
                #  duplicate array values could otherwise allow skipping rows
                #  that might match via larger right resolutions 
                elif dynr and (res_r[row, col] < res_r[row+1, col]):
                    index = row + 1

            skip[row, col] = index

    # Then take the minimum to the left for each element
    # i.e. `skip = np.minimum.accumulate(skip, axis=-1)`
    for row in nb.prange(x):
        minim = np.inf 

        for col in range(y):
            skip[row, col] = minim = min(minim, skip[row, col])

    # If resolution varies, ensure left resolution is handled:
    #  duplicate array values could otherwise allow skipping rows
    #  that might match via larger left resolutions 
    if dynl:
        for col in nb.prange(y):
            index = skip[-1, col-1]
            for row in range(x-2, -1, -1):
                if res_l[row, col] < res_l[row+1, col]:
                    index = row + 1
                skip[row, col-1] = min(index, skip[row, col-1])
    return skip


@nb.njit([
    i32[:,:](fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], boolean[:], boolean, pb_type)
    for fx in INPUT_TYPES
], cache=True, parallel=True, nogil=True)
def bruteforce_original(a1, a2, a1l, a1r, a2l, a2r, all_nan_col, use_locus, progress=None):
    """ Brute-force method to find neighbors within left/right distance tolerance.
    
    Note that this method *requires* a1 and a2 to be lexicographically sorted,
    or else some valid matches may be missed. For our application, this
    constraint can be adhered to for free - so long as care is taken when
    combining grids together into their extended-dimension representation::

        bruteforce(A, B) -> extended-dimension grid AB
        bruteforce(AB, C) -> ABC

    The remaining l/r parameters correspond to the left and right side neighbor
    distance tolerance for a1 and a2, respectively. 

    """
    # Get row/col shape for each array
    r1, c1 = a1.shape
    r2, c2 = a2.shape
    shapes = [a1l.shape, a1r.shape, a2l.shape, a2r.shape]

    # # Sanity check on number of columns across all arrays
    # assert(sum([s[-1] == c1 for s in shapes] + [len(all_nan_col) == c1])), (
    #     f'Arrays must all have the same number of columns: {shapes}')

    # # Sanity check on number of elements over values and left/right resolutions
    # for label,n,l,r in [('len(a1)', r1, a1l, a1r), ('len(a2)', r2, a2l, a2r)]:
    #     assert((len(l) in [n, 1]) and (len(r) in [n, 1])), (
    #         f'Left/Right resolution arrays must have the same length as the ' +
    #         f'values, or length of 1: {label}={n} vs [l={len(l)}, r={len(r)}]')
    
    # Create skip lists, which gives index of next unique value along each axis
    skip1 = create_skiplist(a1, a1l, a1r)
    skip2 = create_skiplist(a2, a2l, a2r)

    # Allow numba to infer the type of the matches array
    match = [(np.int32(x), np.int32(x)) for x in range(0)]

    # Create step sizes to skip over columns containing all NaNs
    steps = np.append(np.cumsum(all_nan_col)[~all_nan_col], 0)

    # Drop any columns that are composed entirely of NaN values
    a1  =  a1[..., ~all_nan_col]
    a2  =  a2[..., ~all_nan_col]
    a1l = a1l[..., ~all_nan_col]
    a1r = a1r[..., ~all_nan_col]
    a2l = a2l[..., ~all_nan_col]
    a2r = a2r[..., ~all_nan_col]

    # Flags indicating whether resolutions are different across rows
    is_dynamic1 = len(a1l) == r1
    is_dynamic2 = len(a2l) == r2

    # Static resolution arrays are vectors of length the number of columns
    a1l_statics = a1l[0]
    a1r_statics = a1r[0]
    a2l_statics = a2l[0]
    a2r_statics = a2r[0]

    # debug = False 
    # l1 = 'ix1'
    # l2 = 'ix2'
    # loop_count = 0 

    # Iterate all rows in a1
    ix1 = start = 0 
    while ix1 < r1:
        # if debug: print(f'\nstart while with start={start}  n_match={(len(match))}')
        b1  =  a1[ix1]
        b1l = a1l[ix1] if is_dynamic1 else a1l_statics
        b1r = a1r[ix1] if is_dynamic1 else a1r_statics

        # Start iterating a2 from the index set during the previous a1 rows
        ix2 = start
        new = False
        # if debug: print(f'\tstarting loop with {l1}={ix1}  {l2}={ix2}  n_match={(len(match))}')

        # Iterate rows in a2
        while ix2 < r2:
            # loop_count += 1
            b2  = a2[ix2]
            b2l = a2l[ix2] if is_dynamic2 else a2l_statics
            b2r = a2r[ix2] if is_dynamic2 else a2r_statics

            invalid_dim, above = check(b1, b1l, b1r, b2, b2l, b2r, use_locus)
            invalid_dim += steps[invalid_dim]
            # if debug:
            #     print(f'\t\t{b1}  vs  {b2}')
            #     print(f'\t\tinvalid dim: {invalid_dim}  b2 above b1: {above == 1}')
            # print(f'{b1}  vs  {b2}')
            # print(f'invalid dim: {invalid_dim}  b2 above b1: {above == 1}')

            # This is only skipping based on the first dimension, but
            # implementing the full dimension skipping logic has a lot of
            # edge cases to handle. For now, the additional speed isn't
            # worth the time required. That may change if much larger number
            # of dimensions needs to be handled (as allowing logic to skip
            # dims other than the first would net larger improvements)
            # if (ix2 == k) and (above == 1):
            # #     # if ix1 > 8000: print(f'\n({ix1}, {ix2}):  {b1}  vs  {b2}')
            #     ix1 = skip1[ix1][invalid_dim] - 1
            #     # if ix1 > 8000: 
            #     #     print(f'invalid dim: {invalid_dim}  b2 above b1: {above == 1}')
            #     #     print(f'breaking; continuing at ix1={ix1+1}  ix2={start}')
            #     #     print(f'current matches: {len(match)}')
            #     # break 

            if (invalid_dim != 0) and not new:
                new   = True
                start = ix2
                # if debug: print(f'\t\tset start = {start}')
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
                    # Note: now handled via extra skip column at index -1
                    # if invalid_dim < 0: 
                        # print(f'\n({ix1}, {ix2}):  {b1}  vs  {b2}')
                        # ix2 = start
                        # b2   = a2[ix2]
                        # b2l = a2l[ix2] if dynamic_r2 else a2ls
                        # b2r = a2r[ix2] if dynamic_r2 else a2rs

                        # invalid_dim, above = check(b1, b1l, b1r, b2, b2l, b2r)
                        # invalid_dim += steps[invalid_dim]
                        # if above == 1:
                        #     ix1 = skip1[ix1][invalid_dim] - 1
                        # elif above == 0:
                            # start = skip2[start][invalid_dim]
                        # print(f'invalid dim: {invalid_dim}  b2 above b1: {above == 1}')
                        # print(f'breaking; continuing at ix1={ix1+1}  ix2={start}')
                        # print(f'current matches: {len(match)}')
                        # if debug: print(f'\t\tbreaking; continuing at {l1}={ix1+1}  {l2}={start}')
                        # break

                # Skip forward based on whichever dimension caused the failure
                # (or the dimension prior to it, if we're above all candidates)
                ix2 = skip2[ix2][invalid_dim]
                # print(f'setting ix2 = {ix2}')
                # if debug: print(f'\t\tsetting {l2} = {ix2}')
            # Store match indices and increment the index
            else: 
                match.append((ix1, ix2))
                # if debug: print(f'\t\t----- match @ {match[-1]}')
                ix2 = ix2 + 1 
        ix1 = ix1 + 1
        # if debug: print(f'\tfinished loop with {l1}={ix1}  {l2}={ix2}  n_match={(len(match))}')
        # if ix1 > 14: 
        #     print(match)
        #     assert(0)
        # if debug: print(f'finished while with start={start}  n_match={(len(match))}')
        if progress is not None: 
            update(progress, ix1)
    # print(f'matches: {len(match)}\n\n')
    # # if len(match) != 10752: assert(0)
    # assert(0)
    # print('loop_count:', loop_count)
    if len(match) == 0:
        return np.empty((0,2), dtype=np.int32)
    return np.array(match, dtype=np.int32)


@nb.njit([
    UniTuple(i32, 3)(DictType(i32, ListType(i32)), i32, i32, UniTuple(fx[:,:],3), UniTuple(fx[:,:],3), i32[:,:], i32[:], boolean, boolean)
    for fx in INPUT_TYPES
], cache=True, nogil=True)
def loop(matches, ix1, start, a1lr, a2lr, skip, steps, swapped, use_locus):
    """ Core bruteforce loop """
    def get_index(i, alr):
        a, l, r = alr
        return (a[i],l[i],r[i]) if len(l) == len(a) else (a[i],l[0],r[0])

    b1, b1l, b1r = get_index(ix1, a1lr)

    # Keep track of item counts
    n_match = 0
    n_loops = 0

    # Start iterating a2 from the index set during the previous a1 rows
    ix2 = start
    new = False

    # Iterate rows in a2
    while ix2 < len(a2lr[0]):
        n_loops += 1
        
        b2, b2l, b2r = get_index(ix2, a2lr)
        invalid_dim, above = check(b1, b1l, b1r, b2, b2l, b2r, use_locus)
        invalid_dim += steps[invalid_dim]

        if (invalid_dim != 0) and not new:
            new   = True
            start = ix2

        # If this isn't a match, skip forward
        if invalid_dim > -1:

            # If the current a2 invalid dimension value is above a1's
            # value, skip forward based on the prior dimension's skip list
            if above:
                invalid_dim -= 1

                # If we can no longer skip forward, move on to the next
                # a1 value and skip back to the first candidate found
                # during this round of a2 iteration
                # Note: now handled via extra skip column at index -1
                # if invalid_dim < 0: break

            # Skip forward based on whichever dimension caused the failure
            # (or the dimension prior to it, if we're above all candidates)
            ix2 = skip[ix2][invalid_dim]

        # Store match indices and increment the index
        else: 
            n_match = n_match + 1 
            k, v    = (ix2, ix1) if swapped else (ix1, ix2)
            ix2     = np.int32(ix2 + 1)
            if k in matches: matches[k].append(v)
            else:            matches[k] = List([np.int32(v)])

    return start, n_match, n_loops


@nb.njit([
    Tuple((UniTuple(fx[:,:], 3), UniTuple(fx[:,:], 3), UniTuple(i32[:,:], 2), i32[:]))(
        fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], boolean[:])
    for fx in INPUT_TYPES
], cache=True, nogil=True)
def bruteforce_setup(a1, a2, a1l, a1r, a2l, a2r, all_nan_col):
    # Get row/col shape for each array
    r1, c1 = a1.shape
    r2, c2 = a2.shape
    shapes = [a1.shape, a2.shape, a1l.shape, a1r.shape, a2l.shape, a2r.shape]

    # # Sanity check on number of columns across all arrays
    # assert(sum([s[-1] == c1 for s in shapes] + [len(all_nan_col) == c1])), (
    #     f'Arrays must all have the same number of columns: {shapes}')

    # # Sanity check on number of elements over values and left/right resolutions
    # for label,n,l,r in [('len(a1)', r1, a1l, a1r), ('len(a2)', r2, a2l, a2r)]:
    #     assert((len(l) in [n, 1]) and (len(r) in [n, 1])), (
    #         f'Left/Right resolution arrays must have the same length as the ' +
    #         f'values, or length of 1: {label}={n} vs [l={len(l)}, r={len(r)}]')
    
    # Create skip lists, which gives index of next unique value along each axis
    skip1 = create_skiplist(a1, a1l, a1r)
    skip2 = create_skiplist(a2, a2l, a2r)

    # Create step sizes to skip over columns containing all NaNs
    steps = np.append(np.cumsum(all_nan_col)[~all_nan_col], 0).astype(np.int32)

    # Drop any columns that are composed entirely of NaN values
    a1  =  a1[..., ~all_nan_col]
    a2  =  a2[..., ~all_nan_col]
    a1l = a1l[..., ~all_nan_col]
    a1r = a1r[..., ~all_nan_col]
    a2l = a2l[..., ~all_nan_col]
    a2r = a2r[..., ~all_nan_col]

    return (a1, a1l, a1r), (a2, a2l, a2r), (skip1, skip2), steps


@nb.njit([
    i32[:,:](fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], boolean[:], boolean, pb_type)
    for fx in INPUT_TYPES
], cache=True, nogil=True)
def bruteforce_single(a1, a2, a1l, a1r, a2l, a2r, all_nan_col, use_locus, progress=None):
    """ Brute-force method to find neighbors within left/right distance tolerance.
    
    Note that this method *requires* a1 and a2 to be lexicographically sorted,
    or else some valid matches may be missed. For our application, this
    constraint can be adhered to for free - so long as care is taken when
    combining grids together into their extended-dimension representation::

        bruteforce(A, B) -> extended-dimension grid AB
        bruteforce(AB, C) -> ABC

    The remaining l/r parameters correspond to the left and right side neighbor
    distance tolerance for a1 and a2, respectively. 

    """
    a1lr, a2lr, (skip1, skip2), steps = bruteforce_setup(
        a1, a2, a1l, a1r, a2l, a2r, all_nan_col)
    
    # Matches are stored as a dict of lists
    matches = Dict.empty(i32, List.empty_list(i32))
    n_match = 0 
    
    # Iterate all rows in a1
    ix1 = start = 0 

    while ix1 < len(a1):
        start, mc, lc = loop(matches, ix1, start, a1lr, a2lr, skip2, steps, False, use_locus)
        n_match = n_match + mc
        ix1 = ix1 + 1

        if progress is not None: 
            update(progress, ix1)

    result = np.empty((2, n_match), dtype=np.int32)
    index  = 0
    for m1 in sorted(matches):
        for m2 in matches[m1]:
            result[0, index] = m1
            result[1, index] = m2
            index += 1
    return result


@nb.njit([
    i32[:,:](fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:,:], boolean[:], boolean, pb_type)
    for fx in INPUT_TYPES
], cache=True, nogil=True)
def bruteforce_double(a1, a2, a1l, a1r, a2l, a2r, all_nan_col, use_locus, progress=None):
    """ Same as bruteforce_single, but adaptively switches back and forth 
        between the two arrays. This speeds up the overall method by allowing
        both arrays to contribute towards skipping forward indices. 
    """
    a1lr, a2lr, (skip1, skip2), steps = bruteforce_setup(a1, a2, a1l, a1r, a2l, a2r, all_nan_col)

    # Iterate all rows in a1
    start1 = 0
    start2 = 0

    reps1 = 1 
    reps2 = 1

    addl = 20
    mult = 1.1

    # Matches are stored as a dict of lists
    matches = Dict.empty(i32, List.empty_list(i32))
    n_match = 0 

    while (start2 < len(a1)) and (start1 < len(a2)):

        n_loop1 = 0
        for _ in range(max(1, int(reps1))):
            if start2 < len(a1):
                start1, mc, lc = loop(matches, start2, start1, a1lr, a2lr, skip2, steps, False, use_locus)
                n_match = n_match + mc
                n_loop1 = n_loop1 + lc
                start2  = start2 + 1
            else: break
        else:

            n_loop2 = 0
            for _ in range(max(1, int(reps2))):
                if start1 < len(a2):
                    start2, mc, lc = loop(matches, start1, start2, a2lr, a1lr, skip1, steps, True, use_locus)
                    n_match = n_match + mc
                    n_loop2 = n_loop2 + lc
                    start1  = start1 + 1
                else: break

            if (n_loop1 / reps1) <= (n_loop2 / reps2):
                reps1 = min(1000, reps1 * mult + addl)
                reps2 = max(1, (reps2 - addl) / mult)
            else: 
                reps2 = min(1000, reps2 * mult + addl)
                reps1 = max(1, (reps1 - addl) / mult)

            if progress is not None: 
                update(progress, start2)

    result = np.empty((2, n_match), dtype=np.int32)
    index  = 0
    for m1 in sorted(matches):
        for m2 in matches[m1]:
            result[0, index] = m1
            result[1, index] = m2
            index += 1
    return result
