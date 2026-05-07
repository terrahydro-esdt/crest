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
from .bruteforce_numba import INPUT_TYPES

import numba as nb
import numpy as np

@nb.njit([UniTuple(fx[:,:], 3)(fx[:,:]) for fx in INPUT_TYPES], 
    cache=True, nogil=True, parallel=False)
def split_array(array):
    n_features = array.shape[1] // 3
    return array[:, :n_features], array[:, n_features:n_features*2], array[:, n_features*2:]


@nb.njit([
    UniTuple(i32, 3)(boolean[:,:], fx[:,:], fx[:,:], fx[:,:], fx[:], fx[:], fx[:], boolean)
    for fx in INPUT_TYPES
], cache=True, nogil=True, parallel=False)
def multicheck(valid_cols, xs, xrls, xrrs, y, yrl, yrr, use_locus):
    """ lambda a,b,c,x,y,z: ((b+a)/2 <= y <= (c+b)/2) or ((y+x)/2 <= b <= (z+y)/2) 
        
        Returns
        -------
        int, int
            - Dimension which caused the check to fail (-1 if the check succeeds)
            - Flag specifying if the failure is due to being above (1) or below (0)

    """
    # count = 0
    for j in range(len(xs)-1,-1,-1):
        x = xs[j]
        xrl = xrls[j]
        xrr = xrrs[j]

        # for k, i in enumerate(order):#range(len(x))):
        for i in range(len(x)):
            if valid_cols[j, i]:
                # if np.isnan(x[i]):
                    # continue
                if use_locus:
                    if xrl[i] > yrr[i]:
                        return j, i, 0
                    if yrl[i] > xrr[i]:
                        return j, i, 1
                else:
                    # count += 1
                    if yrl[i] > x[i]:
                        # count += 1
                        if xrl[i] > y[i]:
                            return j, i, 0#, count
                        # count += 1
                        if y[i] > xrr[i]:
                            return j, i, 1#, count
                    else:
                        # count += 1
                        if x[i] > yrr[i]:
                            # count += 1
                            if xrl[i] > y[i]:
                                return j, i, 0#, count
                            # count += 1
                            if y[i] > xrr[i]:
                                return j, i, 1#, count
    return len(xs)-1, -1, -1#, count
    

debug=False

@nb.njit(
    [
    UniTuple(i32, 3)(
    # Tuple((i32[:], i32, i32))(
        i32[:], 
        fx[:,:], 
        fx[:,:], 
        boolean[:,:],
        i32[:], boolean[:],
        i32[:,:], i32[:], i32, boolean, boolean
        )
    for fx in INPUT_TYPES
], 
cache=True, nogil=True, parallel=False)
def multiloop(matches, b1, a2, valid_cols, starts, flags, 
    skip, steps, maxim, shuffle, use_locus):
    """ Core bruteforce loop """

    # Keep track of item counts
    n_match = 0
    n_loops = 0

    b1a, b1l, b1r = split_array(b1)
    a2a, a2l, a2r = split_array(a2)

    # Start iterating a2 from the index set during the previous a1 rows
    # start_ix = len(b1a)
    ix2 = np.int32(starts[-1])#starts[start_ix])#0
    # ix2 = np.min(starts)
    # new = False
    broke_early = 0
    # flags = np.zeros((len(b1a),), dtype='bool')
    # starts = np.zeros((len(b1a),), dtype=np.int32) + start
    # print(b1.shape, a2.shape, valid_cols, steps)
    
    # if debug:
    #     print('\t\tstarting multiloop for array',len(b1a),'at index', ix2,'/',len(a2))
    # Iterate rows in a2
    while ix2 < len(a2):
        if (maxim > 0) and (n_match > 0) and (n_loops >= maxim):
            broke_early = 1
            break
        n_loops += 1
        
        # Should be able to allow using a new start value if the fail_array
        # is the array immediately prior to the current array (i.e. len(b1)-1)
        fail_array, invalid_dim, above = multicheck(valid_cols, b1a, b1l, b1r, a2a[ix2], a2l[ix2], a2r[ix2], use_locus)
        # invalid_dim += steps[invalid_dim]

        # if (invalid_dim != 0) and not new:
        #     new = True
        #     start = ix2#s[start_ix] = ix2

        if (invalid_dim != 0):
            if not flags[-1]:
                starts[-1] = ix2
                flags[-1] = True

            if invalid_dim == -1:
                # need to set the index for all 
                for i, flag in enumerate(flags):
                    if not flag:
                        if debug: print('\t\t\tarray',i,'| match at index',ix2)
                        starts[i] = max(ix2, starts[i])
                        flags[i] = True

            elif not flags[fail_array]:
                if debug: print('\t\t\tarray',fail_array,'| failure at index',ix2,' above:',above)
                flags[fail_array] = True
                starts[fail_array] = max(ix2, starts[fail_array])

        # If this isn't a match, skip forward
        if invalid_dim > -1:

            # If the current a2 invalid dimension value is above a1's
            # value, skip forward based on the prior dimension's skip list
            if above:
                invalid_dim -= 1
                # invalid_dim = reorder[invalid_dim] - 1
                # If we can no longer skip forward, move on to the next
                # a1 value and skip back to the first candidate found
                # during this round of a2 iteration
                # if invalid_dim < 0: break

            # Skip forward based on whichever dimension caused the failure
            # (or the dimension prior to it, if we're above all candidates)
            ix2 = skip[ix2][steps[invalid_dim]]

        # Store match indices and increment the index
        else: 
            matches[n_match] = ix2
            n_match = n_match + 1 
            ix2     = np.int32(ix2 + 1)
    # if debug: 
    #     print('\t\tFinished multiloop in',n_loops,'loops with',n_match,'matches')
        # print('\t\t\tNew starts:', starts)
        # print('\t\t\t New flags:', flags)
    # print('\t\tdone')
    if shuffle and n_match:
        np.random.shuffle(matches[:n_match])
    return n_match, n_loops, broke_early





# @nb.njit(
#     [i32[:,:](i32, ListType(fx[:,:]), ListType(i32[:,:]))
#     for fx in INPUT_TYPES], 
# cache=True, nogil=True, parallel=False)
# def handle_numba(zero_index, arrays, array_skips):

#     new = False
#     n_features = arrays[0].shape[1] 
#     n_arrays = len(arrays)
#     dtype = arrays[0].dtype
#     index_array = np.zeros((n_arrays,), dtype=np.int32)
#     index_array[0] = zero_index
#     task_array = index_array.copy()

#     tasks = List([task_array[:1]])
#     steps = np.zeros((n_features//3,), dtype=np.int32)
#     max_len = max([len(arr) for arr in arrays])
#     partial_matches = np.zeros((max_len,), dtype=np.int32)

#     a = np.zeros((n_arrays, n_features), dtype=dtype)

#     max_matches = 100
#     matches = np.empty((max_matches, n_arrays), dtype=np.int32)
#     n_match = 0

#     while len(tasks):

#         task = tasks.pop(0)
#         array_index = len(task)
#         task_array[:array_index] = task
#         task_array[array_index:] = index_array[array_index:]

#         for i in range(array_index):
#             a[i] = arrays[i][task_array[i]]


#         row_index = task_array[array_index]
#         swap = False
#         start, n_match2, n_loop = multiloop(partial_matches, a[:array_index], arrays[array_index], row_index, new, array_skips[array_index], steps, swap)

#         # task_array[:array_index] = index_array[:array_index]
#         for m in range(n_match2):
#             task_array[array_index] = partial_matches[m]
#             if (array_index+1) == n_arrays:

#                 if n_match == max_matches:
#                     matches = np.append(matches, np.empty((max_matches*4, n_arrays), dtype=np.int32), axis=0)
#                     max_matches *= 5

#                 matches[n_match] = task_array
#                 n_match += 1
#                 new = True
#             else:
#                 tasks.append(task_array[:array_index+1].copy())

#         index_array[array_index] = start
#     return matches[:n_match]





# @nb.njit(
#     [i32[:,:](ListType(fx[:,:]), ListType(fx[:,:]), ListType(fx[:,:]))
#     for fx in INPUT_TYPES], 
# cache=True, nogil=True, parallel=True)
# def match_numba(arrays, res_l, res_r):
#     # arrays, res_l, and res_r _must_ all have the same number of features

#     array_skips = List([create_skiplist(arr, rl, rr) for arr, rl, rr in zip(arrays, res_l, res_r)])
#     all_arrays = List([np.append(np.append(a, a-rl, axis=1), a+rr, axis=1) for a, rl, rr in zip(arrays, res_l, res_r)])

#     n_arrays = len(arrays)
#     n_elements = len(arrays[0])

#     n_threads = nb.get_num_threads()
#     matches = List([List.empty_list(nb.int32[:,:]) for _ in range(n_threads)])
#     n_match = np.zeros((n_threads,), dtype=np.int32)

#     for i in nb.prange(n_elements):
#         tid = nb.np.ufunc.parallel._get_thread_id()
#         match = handle_numba(i, all_arrays, array_skips)
#         count = len(match)
#         if count:
#             n_match[tid] += count
#             matches[tid].append(match)

#     result = np.empty((np.sum(n_match), n_arrays), dtype=np.int32)
#     index  = np.int32(0)
#     for i in range(n_threads):
#         for match in matches[i]:
#             result[index:index+len(match)] = match
#             index += len(match)

#     return result

@nb.njit([i32[:, :](fx[:, :]) for fx in INPUT_TYPES], 
    cache=True, nogil=True)
def create_skiplist_multi(array_lr):
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

    array, left, right = split_array(array_lr)

    # Include extra last column for skipping on column=-1
    x, y = array.shape
    skip = np.zeros((x, y+1), dtype=np.int32)
    skip[:, y] = x

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
                elif right[row, col] < right[row+1, col]:
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
    for col in nb.prange(y):
        index = skip[-1, col-1]
        for row in range(x-2, -1, -1):
            if left[row, col] > left[row+1, col]:
                index = row + 1
            skip[row, col-1] = min(index, skip[row, col-1])
    return skip





@nb.njit([boolean[:](fx[:, :]) for fx in INPUT_TYPES], 
    cache=True, nogil=True)
def get_finite_cols(array):
    n_col = array.shape[1]
    valid = np.zeros((n_col,), dtype='bool')
    for col in nb.prange(n_col):
        valid[col] = ~(np.isnan(array[:, col]).all())
    return np.append(valid, np.append(valid, valid))



@nb.njit(
    [i32(fx[:,:], fx[:,:], fx)#, ListType(fx[:,:,:]), i32[:])
    for fx in INPUT_TYPES], 
cache=True, nogil=True, parallel=False, fastmath=True)
def compute_total_mean_variance(means, variances, T):
    n_indices, n_candidates = means.shape
    composed_means = np.empty((n_candidates, ), dtype=np.float32)
    composed_vars = np.empty((n_candidates, ), dtype=np.float32)

    for i in range(n_candidates):
        # 1. Current mean/var for choosing i at curr_index
        mu = means[0, i]
        var = variances[0, i]

        # 2. Remaining items and future indices
        # future_items = [j for j in range(n_items) if j != i]

        for j in range(1, n_indices):
            mus = means[j]
            o2s = variances[j]

            # Boltzmann weights
            w = np.exp(-mus / T)
            w /= np.sum(w)

            # Weighted mean
            weighted_mean = np.sum(w * mus)
            mu += weighted_mean

            # Weighted variance (law of total variance)
            mean_diffs = (mus - weighted_mean) ** 2
            weighted_var = np.sum(w * (o2s + mean_diffs))
            var += weighted_var

        composed_means[i] = mu
        composed_vars[i] = var

    stddev = np.sqrt(composed_vars)
    # stddev = np.sqrt(1. / absloc_count[tid, task_size, remain])
    normal = np.abs(np.random.randn(n_candidates))
    greedy = 1
    return np.argmin(composed_means + stddev * normal * greedy)



from numba import njit
import ctypes
import time
# from scipy.ndimage._interpolation import _nd_image
# Access the _PyTime_AsSecondsDouble and _PyTime_GetSystemClock functions from pythonapi
get_system_clock = ctypes.pythonapi._PyTime_GetSystemClock
as_seconds_double = ctypes.pythonapi._PyTime_AsSecondsDouble

# Set the argument types and return types of the functions
get_system_clock.argtypes = []
get_system_clock.restype = ctypes.c_int64

as_seconds_double.argtypes = [ctypes.c_int64]
as_seconds_double.restype = ctypes.c_double

USE_PARALLEL = True
FULL_STARTS = False
PART_STARTS = True
REORDERINGS = True

debug = False
CHECK_TID = 2

from numba.core.config import NUMBA_NUM_THREADS
def multiset_single(coordinates: list[np.ndarray], resolutions: list[np.ndarray], *args, **kwargs):
    # all_nan_col = np.all([np.isnan(a).all(0) for a in coordinates], axis=0)
    # steps = np.append(np.cumsum(all_nan_col)[~all_nan_col], 0).astype(np.int32)
    # order = sorted(np.arange(len(coordinates)), key=lambda i: len(coordinates[i]))
    # coordinates = List([coordinates[i][..., ~all_nan_col] for i in order])
    # resolutions = List([resolutions[i][..., ~all_nan_col] for i in order])
    # order, arrays = zip(*sorted(enumerate(arrays), key=lambda ia: len(ia[1])))
    # entropy = entropy.astype(arrays[0].dtype)

    # lengths = np.stack([len(a) for a in arrays], axis=0)
    # max_len = lengths.max()
    # arrays = np.stack([np.pad(a, [(0,max_len-len(a)), (0,0)]) for a in arrays], axis=0)
    # if USE_PARALLEL: nb.set_num_threads(6)
    # if not catch:
    #     import contextlib
    #     import io
    #     with io.StringIO() as buf, contextlib.redirect_stdout(buf):
    #         return multiset_single_numba(List(arrays))#[:, np.argsort(order)]

    # array_idxs = np.empty((len(coordinates), 2), dtype=np.int32)
    # start = 0
    # for i, a in enumerate(coordinates):
    #     array_lens[i] = a_len = len(a)
    #     array_idxs[i] = [start, start + a_len]
    #     start += a_len
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

    # r = np.vstack([resolutions[i] for i in order])
    # for i in range(len(resolutions)):
        # resolutions[i] = None
    resolutions = None
    # c = np.c_[c, c-r[..., 0], c+r[..., 1]]
    # r = None

    # arrays = np.vstack(arrays)
    # order, coordinates = zip(*sorted(enumerate(coordinates), key=lambda ia: len(ia[1])))
    # resolutions = [resolutions[i] for i in order]
    # coordinates = [np.ascontiguousarray(np.c_[c, c-r[..., 0], c+r[..., 1]]).astype('float32') for c, r in zip(coordinates, resolutions)]
    # resolutions = None

    # n_arrays = len(arrays)
    # n_cols = arrays[0].shape[1]
    # dtype = arrays[0].dtype
    # max_len = array_lens.max()


    if USE_PARALLEL: 
        n_threads = nb.get_num_threads()#int(min(3, max(1, NUMBA_NUM_THREADS//2)))
        chunksize = 0
        if array_lens[0] > (n_threads*2):
            chunksize = 1+array_lens[0] // (n_threads * 8)
        # nb.set_num_threads(n_threads)
        nb.set_parallel_chunksize(chunksize) 
    return multiset_single_numba(c, array_lens, array_idxs)[:, np.argsort(order)]#List(arrays))


@nb.njit(
    [i32[:,:](fx[:,::1], i32[:], i32[:,:], i32, i32, boolean, boolean)#ListType(fx[:,::1]))#, ListType(fx[:,:,:]), i32[:])
    for fx in INPUT_TYPES], 
cache=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def multiset_single_numba(arrays, array_lens, array_idxs, num_samples, seed, shuffle, use_locus):#, resolutions, steps):
    # a = np.arange(10)
    # _nd_image.zoom_shift(a, None, np.array([2.]), a, 0, 4, 0., 0, False)
    # print('shifted:',a)
    if debug: 
        _start_t = get_system_clock()

        t = get_system_clock()
    #     print('start')

    # We can return early if we generate the requested number of samples
    allow_early_stop = num_samples > 0
    np.random.seed(seed)
    n_arrays = len(array_lens)
    n_rows = array_lens[0]
    n_cols = arrays.shape[1]
    n_features = n_cols // 3
    dtype = arrays.dtype
    finite_cols = np.empty((n_arrays, n_cols), dtype='bool')
    # col_orders = np.empty((n_arrays, n_cols), dtype=np.int32)
    # col_reorders = np.empty((n_arrays, n_cols), dtype=np.int32)
    steps = np.empty((n_arrays, n_features+1), dtype=np.int32)

    for i in range(n_arrays):
        finite_cols[i] = get_finite_cols(arrays[array_idxs[i,0]:array_idxs[i,1], :n_features])

        # order_value = entropy[i]
        # order_value[~finite_cols[i, :n_features]] = np.inf
        # order = np.arange(len(order_value))
        # order = np.argsort(order_value)[::-1]
        # col_orders[i] = np.append(order, np.append(order + n_features, order+n_features*2))
        # col_reorders[i] = np.argsort(col_orders[i])
        # finite_cols[i] = finite_cols[i, col_orders[i]]
        # arrays[i] = np.ascontiguousarray(arrays[i][:, col_orders[i]])
        # steps[i] = np.append(np.cumsum(finite_cols[i][col_orders[i]]), 0) - 1
        steps[i] = np.append(np.cumsum(finite_cols[i][:n_features]), 0) - 1
    # print(finite_cols)
    # print(steps)
    # print('here1')
    # array_skips = [create_skiplist_multi(a[:, s][:, r]) for a,s,r in zip(arrays, finite_cols, col_orders)]
    array_skips = [create_skiplist_multi(arrays[array_idxs[i,0]:array_idxs[i,1], s]) for i,s in enumerate(finite_cols)]
    if debug: print('\t\t\t\tInitialize-top:', as_seconds_double(get_system_clock()-t))
    # print(array_skips)
    # print('here2')
    # array_skips = [create_skiplist_multi(a) for a in arrays]
    # col_orders = col_orders[:, :n_features]
    # Use _iget_num_threads to avoid caching issue with dynamic globals 
    if USE_PARALLEL: n_threads = nb.np.ufunc.parallel._iget_num_threads()
    else:            n_threads = 1
    # check = np.zeros(n_threads, dtype='bool')
    # for i in nb.prange(n_threads):
    #     tid = nb.np.ufunc.parallel._iget_thread_id()
    #     check[tid] = True
    # n_threads = check.sum()

    # print('n threads:', n_threads)
    matches = [List.empty_list(nb.int32[:,:]) for _ in range(n_threads)]
    n_match = np.zeros((n_threads,), dtype=np.int32)
    if debug: t = get_system_clock()

    # max_len = max([len(arr) for arr in arrays])
    # index_array = np.zeros((n_threads, n_arrays), dtype=np.int32)
    max_len = array_lens.max()
    partial_matches = np.empty((n_threads, max_len), dtype=np.int32)
    if debug: print('\t\t\t\tInitialize-bot:', as_seconds_double(get_system_clock()-t))

    base_maximum = 1000
    if allow_early_stop:
        base_maximum = 2 * num_samples
    base_maximum += n_rows + 1
    max_matches = np.zeros(n_threads, dtype=np.int32) + base_maximum
    inner_matches = [np.empty((max_matches[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]


    # Keep (per thread) an array_location array which tracks at what index the 
    # original arrays[] array is now located
    #   This should allow adaptively swapping the order of arrays as a thread
    #   processes more zero_indices (with the objective being to minimize n_tasks)

    max_tasks = np.zeros(n_threads, dtype=np.int32) + base_maximum
    tasks_queue = [np.empty((max_tasks[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]
    tasks_order = [np.empty((max_tasks[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]
    tasks_size = [np.empty((max_tasks[0],), dtype=np.int32) for _ in range(n_threads)]
    tasks_swap = [np.empty((n_arrays,), dtype=np.int32) for _ in range(n_threads)]

    # Source arrays to match against the target array
    # source = np.zeros((n_threads, n_arrays, n_cols), dtype=dtype)
    # if debug: 
    #     print('initialize:', as_seconds_double(get_system_clock()-t))
    #     setup = 0
    #     multi = 0
    #     upper = 0
    #     lower = 0

    # print_every = int(max(0.1 * n_rows, 1))
    # ind_t = get_system_clock()
    # starts = [np.zeros((max_len, n_arrays,), dtype=np.int32) for _ in range(n_threads)]
    # starts = np.zeros((n_threads, n_arrays, max_len, n_arrays), dtype=np.int32)

    # starts = [threads, array being controlled, current task index for each array, start value per array]
    # The second axis represents a selection of which array the starts refer to
    # The last axis represents the start value allowed for each array
    # so starts[:, 1] would be the starting values that each array can use when matching against array 1
    # starts[:, 1, :, 0] indicates the starting value for array 1 that array 0 can use
    # starts[:, 4, :, 2] indicates where array 2 can start looking for a match in array 4 

    # So we select: [
    #   1. the thread
    #   2. the current array being matched against (array_index)
    #   3. the data row being used for each of the task arrays
    #   4. the current task arrays
    # ] 
    # Now, the question is whether any starting indices besides array_index-1
    # can actually be used
    if FULL_STARTS:
        starts = np.zeros((n_threads, n_arrays, max_len, n_arrays), dtype=np.int32)
        new_flags = np.zeros((n_threads, n_arrays, max_len, n_arrays), dtype='bool')
    elif PART_STARTS:
        starts = np.zeros((n_threads, n_arrays,), dtype=np.int32)
        new_flags = np.zeros((n_threads, n_arrays,), dtype='bool')
        starts_prefix = np.zeros((n_threads, n_arrays, n_arrays), dtype=np.int32)

    loop_count = np.zeros((n_threads,), dtype=np.int64)
    task_count = np.zeros((n_threads,), dtype=np.int64)
    # copyt = np.zeros((n_threads,), dtype=np.int64)
    # Track number of loops when 'array j follows array i',
    # i.e. follow_loops[i,j] = multiloop([..., array i], array j)

    if REORDERINGS:
        # follow_loops = np.zeros((n_threads, n_arrays, n_arrays), dtype=np.float32)
        # follow_count = np.ones((n_threads, n_arrays, n_arrays), dtype=np.int32)
        # absloc_loops = np.ones((n_threads, n_arrays, n_arrays), dtype=np.int32)
        absloc_count = np.ones((n_threads, n_arrays, n_arrays), dtype=np.int32)
        absloc_means = np.ones((n_threads, n_arrays, n_arrays), dtype=np.float32) 
        absloc_sqdif = np.zeros((n_threads, n_arrays, n_arrays), dtype=np.float32) 

    # print('starting..')

    prior_size = np.zeros((n_threads,), dtype=np.int32)
    first_pick = np.zeros((n_threads,), dtype='bool')
    n_task2 = np.zeros((n_threads,), dtype=np.int32)


    # array_lens = np.empty((n_arrays,), dtype=np.int32)
    # array_padded = np.empty((n_arrays * max_len, n_cols), dtype=dtype)
    # empty = np.empty((0,0), dtype=dtype)
    # for i, a in enumerate(arrays):
    #     array_lens[i] = a_len = len(a)
    #     array_padded[i*max_len:i*max_len + a_len] = a
    #     arrays[i] = empty
    # arrays = None

    if debug: 
        setup = np.zeros((n_threads,), dtype=np.float32)
        multi = np.zeros((n_threads,), dtype=np.float32)
        upper = np.zeros((n_threads,), dtype=np.float32)
        lower = np.zeros((n_threads,), dtype=np.float32)
        a_setter = np.zeros((n_threads,), dtype=np.float32)
        update = np.zeros((n_threads,), dtype=np.float32)
        print('initialize:', as_seconds_double(get_system_clock()-t))

    
    zero_indices = np.arange(n_rows)
    if shuffle:
        np.random.shuffle(zero_indices)

    # Parallel loop over every row of array 0
    for zero_index in nb.prange(n_rows):

        # Seed the current thread's random generator
        np.random.seed(seed + zero_index)
        if allow_early_stop:
            if n_match.sum() >= num_samples:
                continue

        # Get the actual index to use (which is randomized if shuffle=True)
        zero_index = zero_indices[zero_index]
        if debug: t = get_system_clock()
            
        # if (zero_index % print_every) == 0:
        #     print(zero_index,'/',n_rows,':', as_seconds_double(get_system_clock()-ind_t),'seconds')

        # Use _iget_thread_id to avoid caching issue with dynamic globals 
        tid = nb.np.ufunc.parallel._iget_thread_id() if USE_PARALLEL else 0
        # assert(tid < 2), tid
        # print('\n\nstarting',tid,'@',zero_index,'\n\n')
        # index_array = np.zeros((n_arrays,), dtype=np.int32)
        # index_array[0] = zero_index
        # task_array = index_array[tid]

        # Need to reset if running in parallel since zero_index order is random
        if USE_PARALLEL:
            prior_size[tid] = 0

            if PART_STARTS:
                new_flags[tid] = False
                starts_prefix[tid] = 0
                starts[tid] = 0
                
        # Insert the array 0 index into the task queue as the initial task
        tasks_queue[tid][0, 0] = zero_index  # Array 0 coordinate row index
        tasks_order[tid][0, 0] = 0           # Order of arrays composing the task (only contains array 0)
        tasks_size[tid][0] = 1               # Size of the task (only one array in task starting out)
        tasks_left = 1                       # Only one task remaining to begin with

        match_count = 0  # 0 matches found for current zero_index
        n_task = 0       # 0 tasks processed
        
        # part_reorder = np.empty((n_arrays, n_cols), dtype=np.int32)

        if debug: setup[tid] += as_seconds_double(get_system_clock()-t)

        # Change starts to per thread, so that they carry over for each zero_index       
        # starts = np.zeros((n_arrays,), dtype=np.int32)
        # prior_task = np.array([-1], dtype=np.int32)
        # print('start')
        while tasks_left > 0:
            # if tid == CHECK_TID: print('task start', tid)
            # print('tasks_left',tasks_left)
            # print('tasks_size', tasks_size[tid][:3])
            task_count[tid] += 1
            if debug: t = get_system_clock()
            # starts = np.zeros((n_arrays,), dtype=np.int32)
            n_task += 1
            n_task2[tid] += 1
            tasks_left -= 1

            if shuffle and tasks_left:
                # Get a random task index, and swap it with the last task index
                task_index = np.random.randint(0, tasks_left+1)
                
                # Numpy requires the use of a temporary variable to swap
                tasks_swap[tid][0] = tasks_size[tid][tasks_left]
                tasks_size[tid][tasks_left] = tasks_size[tid][task_index] 
                tasks_size[tid][task_index] = tasks_swap[tid][0]
    
                tasks_swap[tid][:] = tasks_order[tid][tasks_left]
                tasks_order[tid][tasks_left] = tasks_order[tid][task_index] 
                tasks_order[tid][task_index] = tasks_swap[tid]
    
                tasks_swap[tid][:] = tasks_queue[tid][tasks_left]
                tasks_queue[tid][tasks_left] = tasks_queue[tid][task_index] 
                tasks_queue[tid][task_index] = tasks_swap[tid]

            # Get info for the current task (last index in queue)
            task_size = tasks_size[tid][tasks_left]
            task_order = tasks_order[tid][tasks_left, :task_size]

            # if max(task_order) >= n_arrays:
            #     print('ERROR:',tid)
            #     print('tasks_left',tasks_left)
                # print('task_size',tasks_size[tid])
                # assert(0)
            # Can't swap array order for multiloop since the left array's 
            #   row is already decided by the task
            # Instead, the most promising array can be chosen as the next
            #   multiloop target - which requires looking at the score for
            #   each of the remaining possible arrays
            # Also need to track the array order per task since the arrays
            #   composing a task are no longer static

            # for i, option in enumerate(np.setdiff1d(full_options, task_order)):
            #     order_scores[tid, i] = (follow_loops[tid, task_size-1, option]
            #                           / follow_count[tid, task_size-1, option])

            # Next target array is chosen from the remaining options by 
            # calculating the average number of loops for each option
            # remain = np.setdiff1d(full_options, task_order)
            # if tid == CHECK_TID: print('task remain',tid)
            if debug: t2 = get_system_clock()
            if REORDERINGS:
                remain = np.ones((n_arrays,), dtype='bool')
                remain[task_order] = False
                remain = np.where(remain)[0] 
                # Maybe select all task_order rather than only -1, and then take the maximum along axis=1 
                # target = remain[np.argmin(follow_loops[tid, task_order[-1], remain]
                #                         / follow_count[tid, task_order[-1], remain])]
                # if tid == CHECK_TID: print('task target',tid)


                unsampled = absloc_count[tid, task_size, remain] <= 1#n_task2[tid]
                if unsampled.any():
                    # select = 0
                    select = np.random.choice(np.where(unsampled)[0])
                    target = remain[select]
                    # print(unsampled, absloc_count, absloc_count[tid, task_size, remain], remain, remain[unsampled], task_size, task_order, target)
                # else:#elif first_pick[tid]:
                #     select = np.argmin(1. / absloc_means[tid, task_size, remain])
                #     target = remain[select]

                    # print(unsampled, target, remain)

                # # Greedy
                # else:
                #     # target = remain[np.argmin(absloc_loops[tid, task_size, remain]
                #     #                         / absloc_count[tid, task_size, remain])]
                #     target = remain[np.argmin(1. / absloc_means[tid, task_size, remain])]


                # # Weighted random sampling
                # else:
                #     # weight = np.cumsum(absloc_count[tid, task_size, remain]
                #     #                  / absloc_loops[tid, task_size, remain])
                #     weight = np.cumsum(1. / absloc_means[tid, task_size, remain])
                #     rindex = weight[-1] * np.random.rand()
                #     select = np.searchsorted(weight, rindex, side='right')
                #     target = remain[select]


                # # Thompson sampling
                # else: 
                #     stddev = np.sqrt((absloc_sqdif[tid, task_size, remain] / 
                #                       absloc_count[tid, task_size, remain] ** 2))
                #     # stddev = np.sqrt(1. / absloc_count[tid, task_size, remain])
                #     normal = np.abs(np.random.randn(len(remain)))
                #     greedy = 0.5
                #     select = np.argmin(absloc_means[tid, task_size, remain] + 
                #                         stddev * normal * greedy)
                #     target = remain[select]
                #     # if target != 1:
                #     #     print(remain, task_order, target)
                #     #     assert(0)

                #     # print(unsampled)
                #     # print(absloc_count[tid])
                #     # print(task_size, remain, target, unsampled)
                #     # print(unsampled[target])
                #     # assert(0)


                # Thompson sampling with future expectations
                else: 

                    select = compute_total_mean_variance(
                        absloc_means[tid, task_size:][:, remain],
                        (absloc_sqdif[tid, task_size:][:, remain] / 
                         absloc_count[tid, task_size:][:, remain] ** 2),
                        0.75)

                    # curr_means = absloc_means[tid, task_size, remain]
                    # curr_varis = (absloc_sqdif[tid, task_size, remain] / 
                    #               absloc_count[tid, task_size, remain] ** 2)

                    # n_rem = len(remain)

                    # if n_rem > 1:
                    #     # q = get_system_clock()
                    #     masks = np.ones((n_rem, 1, n_arrays), dtype='bool')
                    #     masks[..., task_order] = False
                    #     for i, r in enumerate(remain):
                    #         masks[i, :, r] = False

                    #     means = absloc_means[tid, task_size+1:]
                    #     varis = (absloc_sqdif[tid, task_size+1:] / 
                    #              absloc_count[tid, task_size+1:] ** 2)

                    #     # means = np.expand_dims(means, 0)
                    #     # varis = np.expand_dims(varis, 0)
                    #     means = np.broadcast_to(means, (n_rem, n_rem-1, n_arrays))
                    #     varis = np.broadcast_to(varis, (n_rem, n_rem-1, n_arrays))
                    #     # weight = np.empty((n_rem, n_rem, n_arrays), dtype=np.float32)
                    #     # for i in range(n_rem):
                    #     #     weight[i] = np.exp(-means[0] / T) * masks[i]
                    #     # # weight = np.exp(-means / T) * masks 
                    #     # wgtsum = np.sum(weight, axis=-1)
                    #     # print(wgtsum.shape, weight.shape, means.shape)
                    #     # # wgt_ex = np.expand_dims(wgtsum, -1) + 1e-8
                    #     # # print(wgt_ex.shape)
                    #     # # weight = weight / wgt_ex
                    #     # for i in range(n_arrays):
                    #     #     weight[..., i] = weight[..., i] / wgtsum

                    #     # means = np.where(masks, means[None], 1e8)
                    #     # varis = np.where(masks, varis[None], 0.)
                    #     # means = means[None]
                    #     # varis = varis[None] * masks
                    #     # print(means.shape, varis.shape, masks.shape)
                    #     T = 1
                    #     weight = np.exp(-means / T) * masks 
                    #     # weight = weight / (np.sum(weight, axis=-1)[..., None] + 1e-8)
                    #     wgtsum = np.sum(weight, axis=-1) + 1e-8
                    #     for i in range(n_arrays):
                    #         weight[..., i] = weight[..., i] / wgtsum

                    #     weighted_means = np.sum(weight * means, axis=-1)
                    #     # print(means.shape, weighted_means.shape)
                    #     # weighted_means = weighted_means[:, :, None]
                    #     # print(weighted_means.shape)
                    #     # diffs = means - np.expand_dims(weighted_means, -1)
                    #     diffs = np.empty((n_rem, n_rem-1, n_arrays), dtype=np.float32)
                    #     for i in range(n_arrays):
                    #         diffs[:, :, i] = means[:, :, i] - weighted_means
                    #     varis = np.sum(weight * (varis + diffs ** 2), axis=-1)

                    #     # if absloc_count[tid, task_size, remain].sum() > 1000:
                    #     #     print()
                    #     #     print()
                    #     #     print(task_order, remain, task_size)
                    #     #     print(weight, 'weight')
                    #     #     print(means, 'means')
                    #     #     print(weighted_means, 'weighted')
                    #     #     print(varis, 'varis')
                    #     #     print(curr_means, 'curr_mean')
                    #     #     print()
                    #     #     print(absloc_means)
                    #     #     print(absloc_count)
                    #     #     assert(0)
                    #     curr_means += np.sum(weighted_means, axis=1)
                    #     curr_varis += np.sum(varis, axis=1)
                    #     # thomp[tid] += as_seconds_double(get_system_clock()-q)
                    # stddev = np.sqrt(curr_varis)
                    # # stddev = np.sqrt(1. / absloc_count[tid, task_size, remain])
                    # normal = np.abs(np.random.randn(n_rem))
                    # greedy = 1
                    # select = np.argmin(curr_means + stddev * normal * greedy)
                    target = remain[select]


                # target = task_size
                first_pick[tid] = unsampled[select]

                # if tid == 0: print(task_count[tid], remain, target, task_order)
                # if tid == 1: print('target',target, remain, 'curr task:', task_order)
            else:
                target = task_size
                first_pick[tid] = False

            # print(first_sel, target, remain, absloc_count[tid, task_size, remain][target])
            # print(absloc_count, n_task)
            # print(task_size, task_order, tasks_queue[tid][tasks_left, :task_size])
            # print()
            # if n_task >= 2: assert(0)
            # if n_task > 10000:
            #     print('remain', remain)
            #     print('task_order', task_order)
            #     print('target', target)
            #     print(absloc_loops, '-loops')
            #     print(absloc_count, '-count')
            #     print('argmin', np.argmin(absloc_loops[tid, task_size, remain].astype(np.float32)
            #                             / absloc_count[tid, task_size, remain].astype(np.float32)))
            #     print('sel loop', absloc_loops[tid, task_size, remain])
            #     print('sel count', absloc_count[tid, task_size, remain])
            #     assert(0)
            # if debug:
            #     print('\ncurrent task:', tasks_queue[tid][tasks_left, :task_size])
            #     print('\tTask order:',task_order)
            #     print('\tRemaining: ',remain)
            #     print('\tTarget:    ',target)
            #     print('\tDivision:  ',follow_loops[tid, task_order[-1], remain]
            #                         / follow_count[tid, task_order[-1], remain])

            #     print(follow_loops[tid], '-loops')s
            #     print(follow_count[tid], '-count')
            #     t2 = get_system_clock()
            # if tid == CHECK_TID: print('task source',tid, task_order)

            # source[tid, :task_size] = array_padded[:, tasks_queue[tid][tasks_left, :task_size], :][task_order]

            # for i, array_index in enumerate(task_order):#range(task_size):
            #     # if tid == CHECK_TID: 
            #     #     print('\t',i, array_index, source.shape, tasks_left, tasks_queue[tid].shape)
            #     #     print('\t\t', arrays[array_index].shape)
            #     source[tid, i] = arrays[array_index][tasks_queue[tid][tasks_left, i]]
            #     # part_reorder[i] = col_reorders[i]


            # if tid == CHECK_TID: print('done source')
            # for i in range(task_size):
            #     source[i] = arrays[i][tasks_queue[tid][tasks_left, i]]
            # array_left  = task_size-1
            # array_right = task_size
            # indices = (tid, (array_left, array_right), (array_right, array_left))
            # print(follow_loops[indices], follow_count[indices])
            # print(follow_loops[indices] / follow_count[indices])
            
            # # Swap order of arrays
            # if np.argmin(follow_loops[indices] / follow_count[indices]) == 1:
            #     lr_order = (tid, array_right, array_left)

            # else:
            #     lr_order = (tid, array_left, array_right)

            



            if debug: 
                a_setter[tid] += as_seconds_double(get_system_clock()-t2)

                # print('\tPulled starts for task',task,':', start, 'w/ flags',flags)

            if FULL_STARTS:
                task = tasks_queue[tid][tasks_left, :task_size]
                start = np.zeros((task_size,), dtype=np.int32)
                flags = np.zeros((task_size,), dtype='bool')

                for ti, data_row in enumerate(task):
                    start[ti] = starts[tid, target, data_row, task_order[ti]]
                    flags[ti] = new_flags[tid, target, data_row, task_order[ti]]

            elif PART_STARTS:
                # if tasks_queue[tid][tasks_left, 0] == 1) and ()
                # if (absloc_count[tid, task_size, target] < 5):# or (absloc_count[tid, task_size, target] < _abs_lim):
                #     print('\nstart', absloc_count[tid, task_size, target],(prior_size[tid] != task_size), (starts_prefix[tid, target, :task_size-1] != tasks_queue[tid][tasks_left, :task_size-1]).any(), (starts_prefix[tid, target, task_size-1] > tasks_queue[tid][tasks_left, task_size-1]))
                #     print(target, starts, new_flags)
                #     print(starts_prefix)
                #     print(tasks_left)
                #     print(tasks_queue[tid][tasks_left, :task_size])
                #     print(tasks_queue[tid][tasks_left, task_size-1] > starts_prefix[tid, target, task_size-1])
                #     print('done\n')
                # else: assert(0)

                # if (prior_size[tid] != task_size) or (starts_prefix[tid, target, :task_size-1] != tasks_queue[tid][tasks_left, :task_size-1]).any() or (starts_prefix[tid, target, task_size-1] > tasks_queue[tid][tasks_left, task_size-1]):
                if (prior_size[tid] != task_size) or (starts_prefix[tid, target, task_order] != tasks_queue[tid][tasks_left, :task_size]).any():
                    prior_size[tid] = task_size
                    starts_prefix[tid, target, task_order] = tasks_queue[tid][tasks_left, :task_size]
                    starts[tid, target] = 0
                    new_flags[tid, target] = False

                # elif (prior_size[tid] == task_size) and (starts_prefix[tid, target, :task_size-1] == tasks_queue[tid][tasks_left, :task_size-1]).all() and (tasks_queue[tid][tasks_left, task_size-1] > starts_prefix[tid, target, task_size-1]):
                #     new_flags[tid, target] = False
                #     starts_prefix[tid, target, :task_size] = tasks_queue[tid][tasks_left, :task_size]

                order = np.append(task_order, [target])
                start = starts[tid, order]
                flags = new_flags[tid, order]

            else:
                start = np.array([0 for _ in range(n_arrays)], dtype=np.int32) 
                flags = np.array([False for _ in range(n_arrays)], dtype='bool')

            # if tid == CHECK_TID: print('task pre', tid)
            if debug: 
                upper[tid] += as_seconds_double(get_system_clock()-t)
                _multi_t = get_system_clock()
            
            # aidx = task_order * max_len + tasks_queue[tid][tasks_left, :task_size] #np.ravel_multi_index((task_order, tasks_queue[tid][tasks_left, :task_size]), array_padded.shape)
            # aidx = 
            # tidx = target * max_len
            n_match2, n_loop, broke_early = multiloop(
                partial_matches[tid], 
                # source[tid, :task_size],
                arrays[array_idxs[task_order, 0] + tasks_queue[tid][tasks_left, :task_size]],#task_order, tasks_queue[tid][tasks_left, :task_size]], 
                arrays[array_idxs[target, 0]:array_idxs[target, 1]],# arrays[target], 
                finite_cols[task_order] & finite_cols[target], 
                # arrays[target], finite_cols[:task_size] & finite_cols[target], 
                start, flags, 
                array_skips[target], steps[target], 10000 if first_pick[tid] else 0, shuffle, use_locus)#, col_orders[task_size])
            # print('order',task_order, 'target',target, 'queue',tasks_queue[tid][tasks_left, :task_size],'n_match2', n_match2,'n_loop', n_loop, 'broke',broke_early, 'steps',steps[target],'skips', array_skips[target])
            if debug: 
                multi[tid] += as_seconds_double(get_system_clock()-_multi_t)  
                t = get_system_clock()       
            if REORDERINGS:
                score = ((n_match2 / 10)+1) * ((n_loop / 100)+1)
                # score = np.asarray(score).astype(np.float32)[0]
                # if tid == CHECK_TID: print('task mid', tid)
                # if len(remain) > 1:
                #     follow_count[tid, task_order[-1], target] += 1
                #     follow_loops[tid, task_order[-1], target] += n_loop / len(arrays[target])#n_loop / len(arrays[target])
                # absloc_loops[tid, np.argsort(task_order), task_order] += np.uint16(n_loop)
                # absloc_count[tid, np.argsort(task_order), task_order] += 1

                # absloc_loops[tid, task_size, target] += score
                absloc_count[tid, task_size, target] += 1
                delta = score - absloc_means[tid, task_size, target]
                absloc_means[tid, task_size, target] += delta / absloc_count[tid, task_size, target]
                delta2 = score - absloc_means[tid, task_size, target]
                absloc_sqdif[tid, task_size, target] += delta * delta2

                # # Score affects all arrays used, not just the target
                # for i, j in zip(np.argsort(task_order), task_order):
                #     absloc_loops[tid, i, j] += score
                #     absloc_count[tid, i, j] += 1
                    # delta = score - absloc_means[tid, i, j]
                    # absloc_means[tid, i, j] += delta / absloc_count[tid, i, j]
                    # delta2 = score - absloc_means[tid, i, j]
                    # absloc_sqdif[tid, i, j] += delta * delta2


            # if debug: print('\tAdded',n_loop,'/',len(arrays[target]),'to follow_loops @',task_order[-1], target)
            if FULL_STARTS:
                # tsort = np.argsort(task_order)
                for ti, data_row in enumerate(task):
                    starts[tid, target, data_row:, task_order[ti]] = start[ti]
                    new_flags[tid, target, data_row, task_order[ti]] = flags[ti]
            elif PART_STARTS:
                new_flags[tid, target] = flags[-1]
                starts[tid, target] = start[-1]#  np.min(start)
                # if (absloc_count[tid, task_size, target] < 5):
                #     print('new flags', flags, new_flags)
                #     print('starts', start, starts)
                #     print(n_match2, partial_matches[tid][:min(5, n_match2)])
                # for i, j in zip(np.argsort(task_order), task_order):
                #     starts[tid, i] = start[j]

            # starts[tid, :, s_idx] = start

            # if debug: 
            #     # print('\tnew start for array',task_size-1,'from index',s_idx,'forward:', start[task_size-1])
            #     for ai in range(n_arrays):
            #         print('Array',ai,'can start at:')
            #         print(starts[tid, ..., ai],'- starts')
            #         print(new_flags[tid,...,ai].astype(np.int32), '- flags')
            # if first_pick:
            #     print()
            #     print(broke_early, n_match2)
            #     print(starts[tid])
            #     print(new_flags[tid])
            loop_count[tid] += n_loop
            if debug: 
                # n_loops += n_loop
                update[tid] += as_seconds_double(get_system_clock()-t)
                t = get_system_clock()
            if not broke_early and (n_match2 == 0): continue
            # print('first_pick',first_pick, 'n_match',n_match)

            if REORDERINGS and first_pick[tid]:
            #     # print('first selection')
                tasks_left += 1
                n_task2[tid] -= 1

                continue

            if (task_size+1) < n_arrays:
                if (tasks_left + n_match2) >= max_tasks[tid]:
                    additional = 2 * ((tasks_left + n_match2) + max_tasks[tid])
                    tasks_queue[tid] = np.append(tasks_queue[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
                    tasks_order[tid] = np.append(tasks_order[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
                    tasks_size[tid] = np.append(tasks_size[tid], np.empty((additional,), dtype=np.int32), axis=0)
                    max_tasks[tid] += additional

                # Shift existing tasks forward
                # if tid == 1:
                #     print('first 5 tasks:', tasks_order[tid][:min(tasks_left, 5)])
                #     print('sizes:', tasks_size[tid][:min(tasks_left, 5)])

                # # --- Store new tasks at beginning of array ---
                # curr_queue = tasks_queue[tid][tasks_left, :task_size].copy()
                # curr_order = tasks_order[tid][tasks_left, :task_size].copy()
                # # t = get_system_clock()
                # tasks_size[tid][n_match2:tasks_left+n_match2] = tasks_size[tid][:tasks_left].copy()
                # tasks_queue[tid][n_match2:tasks_left+n_match2] = tasks_queue[tid][:tasks_left].copy()
                # tasks_order[tid][n_match2:tasks_left+n_match2] = tasks_order[tid][:tasks_left].copy()
                # # copyt[tid] += get_system_clock()-t
                # tasks_size[tid][:n_match2] = task_size+1
                # tasks_queue[tid][:n_match2, :task_size] = curr_queue#tasks_queue[tid][tasks_left+n_match2, :task_size]
                # tasks_queue[tid][:n_match2, task_size] = partial_matches[tid, :n_match2][::-1]
                # tasks_order[tid][:n_match2, :task_size] = curr_order#tasks_queue[tid][tasks_left+n_match2, :task_size]
                # tasks_order[tid][:n_match2, task_size] = target

                # --- Store new tasks at the end of the array ---
                tasks_size[tid][tasks_left:tasks_left+n_match2] = task_size+1
                tasks_order[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_order[tid][tasks_left, :task_size]
                tasks_order[tid][tasks_left:tasks_left+n_match2, task_size] = target
                tasks_queue[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                tasks_queue[tid][tasks_left:tasks_left+n_match2, task_size] = partial_matches[tid, :n_match2][::-1]

                # tasks_size[tid][tasks_left:tasks_left+n_match2] = task_size+1
                # tasks_order[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_order[tid][tasks_left, :task_size]
                # tasks_order[tid][tasks_left:tasks_left+n_match2, task_size] = target
                # tasks_queue[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                # tasks_queue[tid][tasks_left:tasks_left+n_match2, task_size] = partial_matches[tid, :n_match2][::-1]


                # if tid == 1:
                #     print('Set tasks_order :',n_match2, ',',task_size,'=',target)
                #     for m in range(n_match2+tasks_left):
                #         print('\t\tTask',m,'-',tasks_order[tid][m,:task_size+1], tasks_size[tid][m])
                #     print('curr_queue',curr_queue)
                #     print('curr_order',curr_order)
                #     print('\tAdded',n_match2,'tasks: from',tasks_left,'to',tasks_left+n_match2,'target:',target)

                # if tasks_left == 0:
                #     print(tasks_queue[tid][:n_match2, :task_size+1])
                # if n_match2 > 0: print('added', n_match2, 'tasks with size', task_size+1, 'and', tasks_left,'tasks remaining')

                tasks_left += n_match2
                # if (tasks_left > 10000) and ((tasks_left % 10000) == 0):
                #     print(tasks_left, 'tasks_left')

            
            # Complete match of all arrays; write directly to inner_matches
            else:
                if (match_count + n_match2) >= max_matches[tid]:
                    additional = 2 * ((match_count + n_match2) + max_matches[tid])
                    inner_matches[tid] = np.append(inner_matches[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
                    max_matches[tid] += additional
                # if debug:
                #     print('\tFound',n_match2,'matches using partial task', tasks_queue[tid][tasks_left, :task_size])
                #     print('\t\tMatches:',partial_matches[tid, :n_match2])
                
                # inner_matches[tid][count:count+n_match2, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                # inner_matches[tid][count:count+n_match2, task_size] = partial_matches[tid, :n_match2]
                inner_matches[tid][match_count:match_count+n_match2, task_order] = tasks_queue[tid][tasks_left, :task_size]
                inner_matches[tid][match_count:match_count+n_match2, target] = partial_matches[tid, :n_match2]
                match_count += n_match2
                n_match[tid] += n_match2
            if debug: lower[tid] += as_seconds_double(get_system_clock()-t)

            if allow_early_stop:
                if n_match.sum() >= num_samples:
                    tasks_left = 0
                    
            # if tid == CHECK_TID: print('task end', tid)
        if match_count:
            match = inner_matches[tid][:match_count]
            # n_match[tid] += match_count
            matches[tid].append(match.copy())

        # if debug: 
        #     print('\nFinished zero_index', zero_index)
        #     print('\tsetup',setup)
        #     print('\tupper',upper)
        #     print('\tmulti',multi)
        #     print('\tlower',lower)
        #     print('\tcount',match_count)
        #     print('\ttasks',n_task)
        #     print('\tloops',n_loops)
        #     print('\tset a',a_setter)
        #     # print(starts,'\n\n')

    if debug: 
        t = get_system_clock()

    if debug: 
        # print('\nFinished zero_index', zero_index)
        total = as_seconds_double(get_system_clock()-_start_t)
        print('\t\t\t\tsetup',setup.max(),'/',total)
        print('\t\t\t\tupper',upper.max(),'/',total)
        print('\t\t\t\tmulti',multi.max(),'/',total)
        print('\t\t\t\tupdate',update.max(),'/',total)
        print('\t\t\t\tlower',lower.max(),'/',total)
        # print('\tcount',match_count)
        # print('\t\t\t\ttasks',task_count.mean(),'/',total)
        # print('\t\t\t\tloops',loop_count.mean(),'/',total)
        print('\t\t\t\tset a',a_setter.max(),'/',total)

    # print('Total loops:', np.sum(loop_count)/1000000, 'million')
    # print('Total tasks:', np.sum(task_count)/1000000, 'million')
    # print('Total loops:', np.sum(loop_count))
    # print('Total tasks:', np.sum(task_count))
    # print('Timing:', np.sum(thomp))
    # if REORDERINGS:
    #     print(np.round(absloc_means[0], 2), '-means')
    #     # print(np.round(absloc_sqdif[0], 2), '-M2')
    #     # print(np.round(absloc_loops[0]/absloc_count[0], 2), '-mean2')
    #     print(absloc_count[0], '-count')
        # print(absloc_loops[0], '-score')
        # for tid in range(len(absloc_means)):
        #     vals = absloc_means[tid]#/absloc_count[tid]
        #     best = np.zeros((n_arrays,), dtype=np.int32)
        #     remaining = np.arange(1, n_arrays)

        #     for i in range(1, n_arrays):
        #         remain = np.ones((n_arrays,), dtype='bool')
        #         remain[best[:i]] = False
        #         remain = np.where(remain)[0] 
        #         best[i] = remain[np.argmin(vals[i, remain])]

        #     print('\tThread',tid,'best order:', best)

    # print('copy time:',as_seconds_double(np.sum(copyt)))
    result = np.empty((np.sum(n_match), n_arrays), dtype=np.int32)
    index  = np.int32(0)
    for i in range(n_threads):
        for match in matches[i]:
            result[index:index+len(match)] = match
            index += len(match)

    if debug: print('finalize:', as_seconds_double(get_system_clock()-t))
    if shuffle: 
        np.random.shuffle(result)
    if allow_early_stop:
        result = result[:num_samples]
    # print('\t\t\t\tMulti:',as_seconds_double(np.mean(multi)),'/',as_seconds_double(get_system_clock()-_start_t))
    return result


# USE_PARALLEL = False
# debug = False

# @nb.njit(
#     [i32[:,:](ListType(fx[:,::1]))#, ListType(fx[:,:,:]), i32[:])
#     for fx in INPUT_TYPES], 
# cache=True, nogil=True, parallel=USE_PARALLEL)
# def multiset_single_numba(arrays):#, resolutions, steps):

#     if debug: 
#         t = get_system_clock()
#         print('start')
#     n_arrays = len(arrays)
#     n_rows, n_cols = arrays[0].shape
#     n_features = n_cols // 3
#     dtype = arrays[0].dtype
#     finite_cols = np.empty((n_arrays, n_cols), dtype='bool')
#     # col_orders = np.empty((n_arrays, n_cols), dtype=np.int32)
#     # col_reorders = np.empty((n_arrays, n_cols), dtype=np.int32)
#     steps = np.empty((n_arrays, n_features+1), dtype=np.int32)

#     for i in range(n_arrays):
#         finite_cols[i] = get_finite_cols(arrays[i][:, :n_features])

#         # order_value = entropy[i]
#         # order_value[~finite_cols[i, :n_features]] = np.inf
#         # order = np.arange(len(order_value))
#         # order = np.argsort(order_value)[::-1]
#         # col_orders[i] = np.append(order, np.append(order + n_features, order+n_features*2))
#         # col_reorders[i] = np.argsort(col_orders[i])
#         # finite_cols[i] = finite_cols[i, col_orders[i]]
#         # arrays[i] = np.ascontiguousarray(arrays[i][:, col_orders[i]])
#         # steps[i] = np.append(np.cumsum(finite_cols[i][col_orders[i]]), 0) - 1
#         steps[i] = np.append(np.cumsum(finite_cols[i][:n_features]), 0) - 1

#     # array_skips = [create_skiplist_multi(a[:, s][:, r]) for a,s,r in zip(arrays, finite_cols, col_orders)]
#     array_skips = [create_skiplist_multi(a[:, s]) for a,s in zip(arrays, finite_cols)]

#     # array_skips = [create_skiplist_multi(a) for a in arrays]
#     # col_orders = col_orders[:, :n_features]
#     # Use _iget_num_threads to avoid caching issue with dynamic globals 
#     if USE_PARALLEL: n_threads = nb.np.ufunc.parallel._iget_num_threads()
#     else:            n_threads = 1
#     # check = np.zeros(n_threads, dtype='bool')
#     # for i in nb.prange(n_threads):
#     #     tid = nb.np.ufunc.parallel._iget_thread_id()
#     #     check[tid] = True
#     # n_threads = check.sum()


#     # print('n threads:', n_threads)
#     matches = List([List.empty_list(nb.int32[:,:]) for _ in range(n_threads)])
#     n_match = np.zeros((n_threads,), dtype=np.int32)
#     max_len = max([len(arr) for arr in arrays])
#     # index_array = np.zeros((n_threads, n_arrays), dtype=np.int32)
#     partial_matches = np.zeros((n_threads, max_len), dtype=np.int32)

#     max_matches = np.zeros(n_threads, dtype=np.int32) + 100
#     inner_matches = [np.empty((max_matches[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]


#     # Keep (per thread) an array_location array which tracks at what index the 
#     # original arrays[] array is now located
#     #   This should allow adaptively swapping the order of arrays as a thread
#     #   processes more zero_indices (with the objective being to minimize n_tasks)

#     max_tasks = np.zeros(n_threads, dtype=np.int32) + 100
#     tasks = [np.asfortranarray(np.zeros((max_tasks[0], n_arrays), dtype=np.int32)) for _ in range(n_threads)]
#     task_array_index = [np.zeros((max_tasks[0],), dtype=np.uint16) for _ in range(n_threads)]

#     if debug: 
#         print('initialize:', as_seconds_double(get_system_clock()-t))
#         setup = 0
#         multi = 0
#         upper = 0
#         lower = 0

#     # print_every = int(max(0.05 * n_rows, 1))

#     for zero_index in nb.prange(n_rows):
#         if debug: t = get_system_clock()
#         # if (zero_index % print_every) == 0:
#         #     print(zero_index,'/',n_rows)

#         # Use _iget_thread_id to avoid caching issue with dynamic globals 
#         tid = nb.np.ufunc.parallel._iget_thread_id() if USE_PARALLEL else 0

#         # index_array = np.zeros((n_arrays,), dtype=np.int32)
#         # index_array[0] = zero_index
#         # task_array = index_array[tid]

#         # tasks = List([np.array([zero_index], dtype=np.int32)])
#         tasks[tid][0, 0] = zero_index
#         task_array_index[tid][0] = 1
#         task_count = 1

#         a = np.zeros((n_arrays, n_cols), dtype=dtype)


#         count = 0
#         n_task = 0
#         n_loops = 0
#         popped = 0
#         a_setter = 0
#         # part_reorder = np.empty((n_arrays, n_cols), dtype=np.int32)

#         if debug: setup += as_seconds_double(get_system_clock()-t)

#         # Change starts to per thread, so that they carry over for each zero_index       
#         starts = np.zeros((n_arrays,), dtype=np.int32)

#         while task_count > 0:
#             if debug: t = get_system_clock()

#             n_task += 1
#             if debug: t2 = get_system_clock()
#             task_count -= 1
#             array_index = task_array_index[tid][task_count]


#             if debug: popped += as_seconds_double(get_system_clock()-t2)


#             if debug: t2 = get_system_clock()
#             for i in range(array_index):
#                 a[i] = arrays[i][tasks[tid][task_count, i]]
#                 # part_reorder[i] = col_reorders[i]
#             if debug: a_setter += as_seconds_double(get_system_clock()-t2)
#             if debug: upper += as_seconds_double(get_system_clock()-t)


#             if debug: t = get_system_clock()
#             n_match2, n_loop = multiloop(
#                 partial_matches[tid], 
#                 a[:array_index], 
#                 arrays[array_index], finite_cols[:array_index] & finite_cols[array_index], 
#                 starts,#row_index, 
#                 array_skips[array_index], steps[array_index])#, col_orders[array_index])
#             n_loops += n_loop
#             if debug: multi += as_seconds_double(get_system_clock()-t)
#             # task_array[:array_index] = index_array[:array_index]
#             if debug: t = get_system_clock()

#             # extend tasks array when we reach current maximum, same as inner_matches
            
#             # pre-allocate partial tasks
#             if (array_index+1) < n_arrays:
#                 if (task_count + n_match2) >= max_tasks[tid]:
#                     additional = 2 * ((task_count + n_match2) - max_tasks[tid])
#                     tasks[tid] = np.append(tasks[tid], np.asfortranarray(np.empty((additional, n_arrays), dtype=np.int32)), axis=0)
#                     task_array_index[tid] = np.append(task_array_index[tid], np.empty((additional,), dtype=np.uint16), axis=0)
#                     max_tasks[tid] += additional

#                 task_array_index[tid][task_count:task_count+n_match2] = array_index+1
#                 tasks[tid][task_count:task_count+n_match2, :array_index] = tasks[tid][task_count, :array_index]
#                 tasks[tid][task_count:task_count+n_match2, array_index] = partial_matches[tid, :n_match2]
#                 task_count += n_match2

#             # write directly to inner_matches
#             else:
#                 if (count + n_match2) >= max_matches[tid]:
#                     additional = 2 * ((count + n_match2) - max_matches[tid])
#                     inner_matches[tid] = np.append(inner_matches[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
#                     max_matches[tid] += additional

#                 inner_matches[tid][count:count+n_match2, :array_index] = tasks[tid][task_count, :array_index]
#                 inner_matches[tid][count:count+n_match2, array_index] = partial_matches[tid, :n_match2]
#                 count += n_match2

#             if debug: lower += as_seconds_double(get_system_clock()-t)


#         if count:
#             match = inner_matches[tid][:count]
#             n_match[tid] += count
#             matches[tid].append(match)

#         if debug: 
#             print('\n', zero_index)
#             print('\tsetup',setup)
#             print('\tupper',upper)
#             print('\tmulti',multi)
#             print('\tlower',lower)
#             print('\tcount',count)
#             print('\ttasks',n_task)
#             print('\tloops',n_loops)
#             print('\t  pop',popped)
#             print('\tset a',a_setter)
#             print(starts)

#     if debug: 
#         print('multi', multi)
#         print('total', total)
#         t = get_system_clock()

#     result = np.empty((np.sum(n_match), n_arrays), dtype=np.int32)
#     index  = np.int32(0)
#     for i in range(n_threads):
#         for match in matches[i]:
#             result[index:index+len(match)] = match
#             index += len(match)

#     if debug: print('finalize:', as_seconds_double(get_system_clock()-t))
#     return result















@nb.njit(
    [i32[:,:](fx[:,::1], i32[:], i32[:,:], i32, i32, boolean, boolean)#ListType(fx[:,::1]))#, ListType(fx[:,:,:]), i32[:])
    for fx in INPUT_TYPES], 
cache=True, nogil=True, parallel=USE_PARALLEL, fastmath=True)
def multiset_full_numba(arrays, array_lens, array_idxs, num_samples, seed, shuffle, use_locus):#, resolutions, steps):
    # If we are generating only a limited number of samples, full should be 
    # used since it samples uniformly over all possibilities - but consequently
    # allows the possibility of duplicates if two threads happen to both create
    # the same sample. If all samples are being generated then single should be
    # used since: there is no possibility of duplicate samples; biased sampling
    # is irrelevant (as all possible samples will be created regardless).
    #   single: parallel over 1st array, draws small number of 1st array points
    #   full: parallel over n_threads, uniform over 1st array
    
    # a = np.arange(10)
    # _nd_image.zoom_shift(a, None, np.array([2.]), a, 0, 4, 0., 0, False)
    # print('shifted:',a)
    if debug: 
        _start_t = get_system_clock()

        t = get_system_clock()
    #     print('start')

    # We can return early if we generate the requested number of samples
    allow_early_stop = num_samples > 0
    np.random.seed(seed)
    n_arrays = len(array_lens)
    n_rows = array_lens[0]
    n_cols = arrays.shape[1]
    n_features = n_cols // 3
    dtype = arrays.dtype
    finite_cols = np.empty((n_arrays, n_cols), dtype='bool')
    # col_orders = np.empty((n_arrays, n_cols), dtype=np.int32)
    # col_reorders = np.empty((n_arrays, n_cols), dtype=np.int32)
    steps = np.empty((n_arrays, n_features+1), dtype=np.int32)

    for i in range(n_arrays):
        finite_cols[i] = get_finite_cols(arrays[array_idxs[i,0]:array_idxs[i,1], :n_features])

        # order_value = entropy[i]
        # order_value[~finite_cols[i, :n_features]] = np.inf
        # order = np.arange(len(order_value))
        # order = np.argsort(order_value)[::-1]
        # col_orders[i] = np.append(order, np.append(order + n_features, order+n_features*2))
        # col_reorders[i] = np.argsort(col_orders[i])
        # finite_cols[i] = finite_cols[i, col_orders[i]]
        # arrays[i] = np.ascontiguousarray(arrays[i][:, col_orders[i]])
        # steps[i] = np.append(np.cumsum(finite_cols[i][col_orders[i]]), 0) - 1
        steps[i] = np.append(np.cumsum(finite_cols[i][:n_features]), 0) - 1
    # print(finite_cols)
    # print(steps)
    # print('here1')
    # array_skips = [create_skiplist_multi(a[:, s][:, r]) for a,s,r in zip(arrays, finite_cols, col_orders)]
    array_skips = [create_skiplist_multi(arrays[array_idxs[i,0]:array_idxs[i,1], s]) for i,s in enumerate(finite_cols)]
    if debug: print('\t\t\t\tInitialize-top:', as_seconds_double(get_system_clock()-t))
    # print(array_skips)
    # print('here2')
    # array_skips = [create_skiplist_multi(a) for a in arrays]
    # col_orders = col_orders[:, :n_features]
    # Use _iget_num_threads to avoid caching issue with dynamic globals 
    if USE_PARALLEL: n_threads = nb.np.ufunc.parallel._iget_num_threads()
    else:            n_threads = 1
    # check = np.zeros(n_threads, dtype='bool')
    # for i in nb.prange(n_threads):
    #     tid = nb.np.ufunc.parallel._iget_thread_id()
    #     check[tid] = True
    # n_threads = check.sum()

    # print('n threads:', n_threads)
    matches = [List.empty_list(nb.int32[:,:]) for _ in range(n_threads)]
    n_match = np.zeros((n_threads,), dtype=np.int32)
    if debug: t = get_system_clock()

    # max_len = max([len(arr) for arr in arrays])
    # index_array = np.zeros((n_threads, n_arrays), dtype=np.int32)
    max_len = array_lens.max()
    partial_matches = np.empty((n_threads, max_len), dtype=np.int32)
    if debug: print('\t\t\t\tInitialize-bot:', as_seconds_double(get_system_clock()-t))

    base_maximum = 1000
    if allow_early_stop:
        base_maximum = 2 * num_samples
    base_maximum += n_rows + 1
    max_matches = np.zeros(n_threads, dtype=np.int32) + base_maximum
    inner_matches = [np.empty((max_matches[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]


    # Keep (per thread) an array_location array which tracks at what index the 
    # original arrays[] array is now located
    #   This should allow adaptively swapping the order of arrays as a thread
    #   processes more zero_indices (with the objective being to minimize n_tasks)

    max_tasks = np.zeros(n_threads, dtype=np.int32) + base_maximum
    tasks_queue = [np.empty((max_tasks[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]
    tasks_order = [np.empty((max_tasks[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]
    tasks_size = [np.empty((max_tasks[0],), dtype=np.int32) for _ in range(n_threads)]
    tasks_swap = [np.empty((n_arrays,), dtype=np.int32) for _ in range(n_threads)]

    # Source arrays to match against the target array
    # source = np.zeros((n_threads, n_arrays, n_cols), dtype=dtype)
    # if debug: 
    #     print('initialize:', as_seconds_double(get_system_clock()-t))
    #     setup = 0
    #     multi = 0
    #     upper = 0
    #     lower = 0

    # print_every = int(max(0.1 * n_rows, 1))
    # ind_t = get_system_clock()
    # starts = [np.zeros((max_len, n_arrays,), dtype=np.int32) for _ in range(n_threads)]
    # starts = np.zeros((n_threads, n_arrays, max_len, n_arrays), dtype=np.int32)

    # starts = [threads, array being controlled, current task index for each array, start value per array]
    # The second axis represents a selection of which array the starts refer to
    # The last axis represents the start value allowed for each array
    # so starts[:, 1] would be the starting values that each array can use when matching against array 1
    # starts[:, 1, :, 0] indicates the starting value for array 1 that array 0 can use
    # starts[:, 4, :, 2] indicates where array 2 can start looking for a match in array 4 

    # So we select: [
    #   1. the thread
    #   2. the current array being matched against (array_index)
    #   3. the data row being used for each of the task arrays
    #   4. the current task arrays
    # ] 
    # Now, the question is whether any starting indices besides array_index-1
    # can actually be used
    if FULL_STARTS:
        starts = np.zeros((n_threads, n_arrays, max_len, n_arrays), dtype=np.int32)
        new_flags = np.zeros((n_threads, n_arrays, max_len, n_arrays), dtype='bool')
    elif PART_STARTS:
        starts = np.zeros((n_threads, n_arrays,), dtype=np.int32)
        new_flags = np.zeros((n_threads, n_arrays,), dtype='bool')
        starts_prefix = np.zeros((n_threads, n_arrays, n_arrays), dtype=np.int32)

    loop_count = np.zeros((n_threads,), dtype=np.int64)
    task_count = np.zeros((n_threads,), dtype=np.int64)
    # copyt = np.zeros((n_threads,), dtype=np.int64)
    # Track number of loops when 'array j follows array i',
    # i.e. follow_loops[i,j] = multiloop([..., array i], array j)

    if REORDERINGS:
        # follow_loops = np.zeros((n_threads, n_arrays, n_arrays), dtype=np.float32)
        # follow_count = np.ones((n_threads, n_arrays, n_arrays), dtype=np.int32)
        # absloc_loops = np.ones((n_threads, n_arrays, n_arrays), dtype=np.int32)
        absloc_count = np.ones((n_threads, n_arrays, n_arrays), dtype=np.int32)
        absloc_means = np.ones((n_threads, n_arrays, n_arrays), dtype=np.float32) 
        absloc_sqdif = np.zeros((n_threads, n_arrays, n_arrays), dtype=np.float32) 

    # print('starting..')

    prior_size = np.zeros((n_threads,), dtype=np.int32)
    first_pick = np.zeros((n_threads,), dtype='bool')
    n_task2 = np.zeros((n_threads,), dtype=np.int32)


    # array_lens = np.empty((n_arrays,), dtype=np.int32)
    # array_padded = np.empty((n_arrays * max_len, n_cols), dtype=dtype)
    # empty = np.empty((0,0), dtype=dtype)
    # for i, a in enumerate(arrays):
    #     array_lens[i] = a_len = len(a)
    #     array_padded[i*max_len:i*max_len + a_len] = a
    #     arrays[i] = empty
    # arrays = None

    if debug: 
        setup = np.zeros((n_threads,), dtype=np.float32)
        multi = np.zeros((n_threads,), dtype=np.float32)
        upper = np.zeros((n_threads,), dtype=np.float32)
        lower = np.zeros((n_threads,), dtype=np.float32)
        a_setter = np.zeros((n_threads,), dtype=np.float32)
        update = np.zeros((n_threads,), dtype=np.float32)
        print('initialize:', as_seconds_double(get_system_clock()-t))

    per_thread = n_rows // n_threads  
    zero_indices = np.arange(n_rows)
    if shuffle:
        np.random.shuffle(zero_indices)

    # Parallel loop over every row of array 0
    for zero_index in nb.prange(n_threads):

        # Seed the current thread's random generator
        np.random.seed(seed + zero_index)
        if allow_early_stop:
            if n_match.sum() >= num_samples:
                continue

        # Get the actual index to use (which is randomized if shuffle=True)
        # zero_index = zero_indices[zero_index]
        if debug: t = get_system_clock()
            
        # if (zero_index % print_every) == 0:
        #     print(zero_index,'/',n_rows,':', as_seconds_double(get_system_clock()-ind_t),'seconds')

        # Use _iget_thread_id to avoid caching issue with dynamic globals 
        tid = nb.np.ufunc.parallel._iget_thread_id() if USE_PARALLEL else 0
        # assert(tid < 2), tid
        # print('\n\nstarting',tid,'@',zero_index,'\n\n')
        # index_array = np.zeros((n_arrays,), dtype=np.int32)
        # index_array[0] = zero_index
        # task_array = index_array[tid]

        # Need to reset if running in parallel since zero_index order is random
        if USE_PARALLEL:
            prior_size[tid] = 0

            if PART_STARTS:
                new_flags[tid] = False
                starts_prefix[tid] = 0
                starts[tid] = 0
        if zero_index == (n_threads-1):
            thread_indices = zero_indices[per_thread * zero_index:].copy()
        else:
            thread_indices = zero_indices[per_thread * zero_index: per_thread * (zero_index+1)].copy()
        t_rows = len(thread_indices)
        np.random.shuffle(thread_indices)

        # Insert the array 0 index into the task queue as the initial task
        tasks_queue[tid][:t_rows, 0] = thread_indices  # Array 0 coordinate row index
        tasks_order[tid][:t_rows, 0] = 0           # Order of arrays composing the task (only contains array 0)
        tasks_size[tid][:t_rows] = 1               # Size of the task (only one array in task starting out)
        tasks_left = t_rows                       # Only one task remaining to begin with

        match_count = 0  # 0 matches found for current zero_index
        n_task = 0       # 0 tasks processed
        
        # part_reorder = np.empty((n_arrays, n_cols), dtype=np.int32)

        if debug: setup[tid] += as_seconds_double(get_system_clock()-t)

        # Change starts to per thread, so that they carry over for each zero_index       
        # starts = np.zeros((n_arrays,), dtype=np.int32)
        # prior_task = np.array([-1], dtype=np.int32)
        # print('start')
        while tasks_left > 0:
            # if tid == CHECK_TID: print('task start', tid)
            # print('tasks_left',tasks_left)
            # print('tasks_size', tasks_size[tid][:3])
            task_count[tid] += 1
            if debug: t = get_system_clock()
            # starts = np.zeros((n_arrays,), dtype=np.int32)
            n_task += 1
            n_task2[tid] += 1
            tasks_left -= 1

            if shuffle and tasks_left:
                # Get a random task index, and swap it with the last task index
                task_index = np.random.randint(0, tasks_left+1)
                
                # Numpy requires the use of a temporary variable to swap
                tasks_swap[tid][0] = tasks_size[tid][tasks_left]
                tasks_size[tid][tasks_left] = tasks_size[tid][task_index] 
                tasks_size[tid][task_index] = tasks_swap[tid][0]
    
                tasks_swap[tid][:] = tasks_order[tid][tasks_left]
                tasks_order[tid][tasks_left] = tasks_order[tid][task_index] 
                tasks_order[tid][task_index] = tasks_swap[tid]
    
                tasks_swap[tid][:] = tasks_queue[tid][tasks_left]
                tasks_queue[tid][tasks_left] = tasks_queue[tid][task_index] 
                tasks_queue[tid][task_index] = tasks_swap[tid]

            # Get info for the current task (last index in queue)
            task_size = tasks_size[tid][tasks_left]
            task_order = tasks_order[tid][tasks_left, :task_size]

            # if max(task_order) >= n_arrays:
            #     print('ERROR:',tid)
            #     print('tasks_left',tasks_left)
                # print('task_size',tasks_size[tid])
                # assert(0)
            # Can't swap array order for multiloop since the left array's 
            #   row is already decided by the task
            # Instead, the most promising array can be chosen as the next
            #   multiloop target - which requires looking at the score for
            #   each of the remaining possible arrays
            # Also need to track the array order per task since the arrays
            #   composing a task are no longer static

            # for i, option in enumerate(np.setdiff1d(full_options, task_order)):
            #     order_scores[tid, i] = (follow_loops[tid, task_size-1, option]
            #                           / follow_count[tid, task_size-1, option])

            # Next target array is chosen from the remaining options by 
            # calculating the average number of loops for each option
            # remain = np.setdiff1d(full_options, task_order)
            # if tid == CHECK_TID: print('task remain',tid)
            if debug: t2 = get_system_clock()
            if REORDERINGS:
                remain = np.ones((n_arrays,), dtype='bool')
                remain[task_order] = False
                remain = np.where(remain)[0] 
                # Maybe select all task_order rather than only -1, and then take the maximum along axis=1 
                # target = remain[np.argmin(follow_loops[tid, task_order[-1], remain]
                #                         / follow_count[tid, task_order[-1], remain])]
                # if tid == CHECK_TID: print('task target',tid)


                unsampled = absloc_count[tid, task_size, remain] <= 1#n_task2[tid]
                if unsampled.any():
                    # select = 0
                    select = np.random.choice(np.where(unsampled)[0])
                    target = remain[select]
                    # print(unsampled, absloc_count, absloc_count[tid, task_size, remain], remain, remain[unsampled], task_size, task_order, target)
                # else:#elif first_pick[tid]:
                #     select = np.argmin(1. / absloc_means[tid, task_size, remain])
                #     target = remain[select]

                    # print(unsampled, target, remain)

                # # Greedy
                # else:
                #     # target = remain[np.argmin(absloc_loops[tid, task_size, remain]
                #     #                         / absloc_count[tid, task_size, remain])]
                #     target = remain[np.argmin(1. / absloc_means[tid, task_size, remain])]


                # # Weighted random sampling
                # else:
                #     # weight = np.cumsum(absloc_count[tid, task_size, remain]
                #     #                  / absloc_loops[tid, task_size, remain])
                #     weight = np.cumsum(1. / absloc_means[tid, task_size, remain])
                #     rindex = weight[-1] * np.random.rand()
                #     select = np.searchsorted(weight, rindex, side='right')
                #     target = remain[select]


                # # Thompson sampling
                # else: 
                #     stddev = np.sqrt((absloc_sqdif[tid, task_size, remain] / 
                #                       absloc_count[tid, task_size, remain] ** 2))
                #     # stddev = np.sqrt(1. / absloc_count[tid, task_size, remain])
                #     normal = np.abs(np.random.randn(len(remain)))
                #     greedy = 0.5
                #     select = np.argmin(absloc_means[tid, task_size, remain] + 
                #                         stddev * normal * greedy)
                #     target = remain[select]
                #     # if target != 1:
                #     #     print(remain, task_order, target)
                #     #     assert(0)

                #     # print(unsampled)
                #     # print(absloc_count[tid])
                #     # print(task_size, remain, target, unsampled)
                #     # print(unsampled[target])
                #     # assert(0)


                # Thompson sampling with future expectations
                else: 

                    select = compute_total_mean_variance(
                        absloc_means[tid, task_size:][:, remain],
                        (absloc_sqdif[tid, task_size:][:, remain] / 
                         absloc_count[tid, task_size:][:, remain] ** 2),
                        0.75)

                    # curr_means = absloc_means[tid, task_size, remain]
                    # curr_varis = (absloc_sqdif[tid, task_size, remain] / 
                    #               absloc_count[tid, task_size, remain] ** 2)

                    # n_rem = len(remain)

                    # if n_rem > 1:
                    #     # q = get_system_clock()
                    #     masks = np.ones((n_rem, 1, n_arrays), dtype='bool')
                    #     masks[..., task_order] = False
                    #     for i, r in enumerate(remain):
                    #         masks[i, :, r] = False

                    #     means = absloc_means[tid, task_size+1:]
                    #     varis = (absloc_sqdif[tid, task_size+1:] / 
                    #              absloc_count[tid, task_size+1:] ** 2)

                    #     # means = np.expand_dims(means, 0)
                    #     # varis = np.expand_dims(varis, 0)
                    #     means = np.broadcast_to(means, (n_rem, n_rem-1, n_arrays))
                    #     varis = np.broadcast_to(varis, (n_rem, n_rem-1, n_arrays))
                    #     # weight = np.empty((n_rem, n_rem, n_arrays), dtype=np.float32)
                    #     # for i in range(n_rem):
                    #     #     weight[i] = np.exp(-means[0] / T) * masks[i]
                    #     # # weight = np.exp(-means / T) * masks 
                    #     # wgtsum = np.sum(weight, axis=-1)
                    #     # print(wgtsum.shape, weight.shape, means.shape)
                    #     # # wgt_ex = np.expand_dims(wgtsum, -1) + 1e-8
                    #     # # print(wgt_ex.shape)
                    #     # # weight = weight / wgt_ex
                    #     # for i in range(n_arrays):
                    #     #     weight[..., i] = weight[..., i] / wgtsum

                    #     # means = np.where(masks, means[None], 1e8)
                    #     # varis = np.where(masks, varis[None], 0.)
                    #     # means = means[None]
                    #     # varis = varis[None] * masks
                    #     # print(means.shape, varis.shape, masks.shape)
                    #     T = 1
                    #     weight = np.exp(-means / T) * masks 
                    #     # weight = weight / (np.sum(weight, axis=-1)[..., None] + 1e-8)
                    #     wgtsum = np.sum(weight, axis=-1) + 1e-8
                    #     for i in range(n_arrays):
                    #         weight[..., i] = weight[..., i] / wgtsum

                    #     weighted_means = np.sum(weight * means, axis=-1)
                    #     # print(means.shape, weighted_means.shape)
                    #     # weighted_means = weighted_means[:, :, None]
                    #     # print(weighted_means.shape)
                    #     # diffs = means - np.expand_dims(weighted_means, -1)
                    #     diffs = np.empty((n_rem, n_rem-1, n_arrays), dtype=np.float32)
                    #     for i in range(n_arrays):
                    #         diffs[:, :, i] = means[:, :, i] - weighted_means
                    #     varis = np.sum(weight * (varis + diffs ** 2), axis=-1)

                    #     # if absloc_count[tid, task_size, remain].sum() > 1000:
                    #     #     print()
                    #     #     print()
                    #     #     print(task_order, remain, task_size)
                    #     #     print(weight, 'weight')
                    #     #     print(means, 'means')
                    #     #     print(weighted_means, 'weighted')
                    #     #     print(varis, 'varis')
                    #     #     print(curr_means, 'curr_mean')
                    #     #     print()
                    #     #     print(absloc_means)
                    #     #     print(absloc_count)
                    #     #     assert(0)
                    #     curr_means += np.sum(weighted_means, axis=1)
                    #     curr_varis += np.sum(varis, axis=1)
                    #     # thomp[tid] += as_seconds_double(get_system_clock()-q)
                    # stddev = np.sqrt(curr_varis)
                    # # stddev = np.sqrt(1. / absloc_count[tid, task_size, remain])
                    # normal = np.abs(np.random.randn(n_rem))
                    # greedy = 1
                    # select = np.argmin(curr_means + stddev * normal * greedy)
                    target = remain[select]


                # target = task_size
                first_pick[tid] = unsampled[select]

                # if tid == 0: print(task_count[tid], remain, target, task_order)
                # if tid == 1: print('target',target, remain, 'curr task:', task_order)
            else:
                target = task_size
                first_pick[tid] = False

            # print(first_sel, target, remain, absloc_count[tid, task_size, remain][target])
            # print(absloc_count, n_task)
            # print(task_size, task_order, tasks_queue[tid][tasks_left, :task_size])
            # print()
            # if n_task >= 2: assert(0)
            # if n_task > 10000:
            #     print('remain', remain)
            #     print('task_order', task_order)
            #     print('target', target)
            #     print(absloc_loops, '-loops')
            #     print(absloc_count, '-count')
            #     print('argmin', np.argmin(absloc_loops[tid, task_size, remain].astype(np.float32)
            #                             / absloc_count[tid, task_size, remain].astype(np.float32)))
            #     print('sel loop', absloc_loops[tid, task_size, remain])
            #     print('sel count', absloc_count[tid, task_size, remain])
            #     assert(0)
            # if debug:
            #     print('\ncurrent task:', tasks_queue[tid][tasks_left, :task_size])
            #     print('\tTask order:',task_order)
            #     print('\tRemaining: ',remain)
            #     print('\tTarget:    ',target)
            #     print('\tDivision:  ',follow_loops[tid, task_order[-1], remain]
            #                         / follow_count[tid, task_order[-1], remain])

            #     print(follow_loops[tid], '-loops')s
            #     print(follow_count[tid], '-count')
            #     t2 = get_system_clock()
            # if tid == CHECK_TID: print('task source',tid, task_order)

            # source[tid, :task_size] = array_padded[:, tasks_queue[tid][tasks_left, :task_size], :][task_order]

            # for i, array_index in enumerate(task_order):#range(task_size):
            #     # if tid == CHECK_TID: 
            #     #     print('\t',i, array_index, source.shape, tasks_left, tasks_queue[tid].shape)
            #     #     print('\t\t', arrays[array_index].shape)
            #     source[tid, i] = arrays[array_index][tasks_queue[tid][tasks_left, i]]
            #     # part_reorder[i] = col_reorders[i]


            # if tid == CHECK_TID: print('done source')
            # for i in range(task_size):
            #     source[i] = arrays[i][tasks_queue[tid][tasks_left, i]]
            # array_left  = task_size-1
            # array_right = task_size
            # indices = (tid, (array_left, array_right), (array_right, array_left))
            # print(follow_loops[indices], follow_count[indices])
            # print(follow_loops[indices] / follow_count[indices])
            
            # # Swap order of arrays
            # if np.argmin(follow_loops[indices] / follow_count[indices]) == 1:
            #     lr_order = (tid, array_right, array_left)

            # else:
            #     lr_order = (tid, array_left, array_right)

            



            if debug: 
                a_setter[tid] += as_seconds_double(get_system_clock()-t2)

                # print('\tPulled starts for task',task,':', start, 'w/ flags',flags)

            if FULL_STARTS:
                task = tasks_queue[tid][tasks_left, :task_size]
                start = np.zeros((task_size,), dtype=np.int32)
                flags = np.zeros((task_size,), dtype='bool')

                for ti, data_row in enumerate(task):
                    start[ti] = starts[tid, target, data_row, task_order[ti]]
                    flags[ti] = new_flags[tid, target, data_row, task_order[ti]]

            elif PART_STARTS:
                # if tasks_queue[tid][tasks_left, 0] == 1) and ()
                # if (absloc_count[tid, task_size, target] < 5):# or (absloc_count[tid, task_size, target] < _abs_lim):
                #     print('\nstart', absloc_count[tid, task_size, target],(prior_size[tid] != task_size), (starts_prefix[tid, target, :task_size-1] != tasks_queue[tid][tasks_left, :task_size-1]).any(), (starts_prefix[tid, target, task_size-1] > tasks_queue[tid][tasks_left, task_size-1]))
                #     print(target, starts, new_flags)
                #     print(starts_prefix)
                #     print(tasks_left)
                #     print(tasks_queue[tid][tasks_left, :task_size])
                #     print(tasks_queue[tid][tasks_left, task_size-1] > starts_prefix[tid, target, task_size-1])
                #     print('done\n')
                # else: assert(0)

                # if (prior_size[tid] != task_size) or (starts_prefix[tid, target, :task_size-1] != tasks_queue[tid][tasks_left, :task_size-1]).any() or (starts_prefix[tid, target, task_size-1] > tasks_queue[tid][tasks_left, task_size-1]):
                if (prior_size[tid] != task_size) or (starts_prefix[tid, target, task_order] != tasks_queue[tid][tasks_left, :task_size]).any():
                    prior_size[tid] = task_size
                    # starts_prefix[tid, target, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                    starts_prefix[tid, target, task_order] = tasks_queue[tid][tasks_left, :task_size]
                    starts[tid, target] = 0
                    new_flags[tid, target] = False

                # elif (prior_size[tid] == task_size) and (starts_prefix[tid, target, :task_size-1] == tasks_queue[tid][tasks_left, :task_size-1]).all() and (tasks_queue[tid][tasks_left, task_size-1] > starts_prefix[tid, target, task_size-1]):
                #     new_flags[tid, target] = False
                #     starts_prefix[tid, target, :task_size] = tasks_queue[tid][tasks_left, :task_size]

                order = np.append(task_order, [target])
                start = starts[tid, order]
                flags = new_flags[tid, order]

            else:
                start = np.array([0 for _ in range(n_arrays)], dtype=np.int32) 
                flags = np.array([False for _ in range(n_arrays)], dtype='bool')

            # if tid == CHECK_TID: print('task pre', tid)
            if debug: 
                upper[tid] += as_seconds_double(get_system_clock()-t)
                _multi_t = get_system_clock()
            
            # aidx = task_order * max_len + tasks_queue[tid][tasks_left, :task_size] #np.ravel_multi_index((task_order, tasks_queue[tid][tasks_left, :task_size]), array_padded.shape)
            # aidx = 
            # tidx = target * max_len
            n_match2, n_loop, broke_early = multiloop(
                partial_matches[tid], 
                # source[tid, :task_size],
                arrays[array_idxs[task_order, 0] + tasks_queue[tid][tasks_left, :task_size]],#task_order, tasks_queue[tid][tasks_left, :task_size]], 
                arrays[array_idxs[target, 0]:array_idxs[target, 1]],# arrays[target], 
                finite_cols[task_order] & finite_cols[target], 
                # arrays[target], finite_cols[:task_size] & finite_cols[target], 
                start, flags, 
                array_skips[target], steps[target], 10000 if first_pick[tid] else 0, shuffle, use_locus)#, col_orders[task_size])
            if debug: 
                multi[tid] += as_seconds_double(get_system_clock()-_multi_t)  
                t = get_system_clock()       
            if REORDERINGS:
                score = ((n_match2 / 10)+1) * ((n_loop / 100)+1)
                # score = np.asarray(score).astype(np.float32)[0]
                # if tid == CHECK_TID: print('task mid', tid)
                # if len(remain) > 1:
                #     follow_count[tid, task_order[-1], target] += 1
                #     follow_loops[tid, task_order[-1], target] += n_loop / len(arrays[target])#n_loop / len(arrays[target])
                # absloc_loops[tid, np.argsort(task_order), task_order] += np.uint16(n_loop)
                # absloc_count[tid, np.argsort(task_order), task_order] += 1

                # absloc_loops[tid, task_size, target] += score
                absloc_count[tid, task_size, target] += 1
                delta = score - absloc_means[tid, task_size, target]
                absloc_means[tid, task_size, target] += delta / absloc_count[tid, task_size, target]
                delta2 = score - absloc_means[tid, task_size, target]
                absloc_sqdif[tid, task_size, target] += delta * delta2

                # # Score affects all arrays used, not just the target
                # for i, j in zip(np.argsort(task_order), task_order):
                #     absloc_loops[tid, i, j] += score
                #     absloc_count[tid, i, j] += 1
                    # delta = score - absloc_means[tid, i, j]
                    # absloc_means[tid, i, j] += delta / absloc_count[tid, i, j]
                    # delta2 = score - absloc_means[tid, i, j]
                    # absloc_sqdif[tid, i, j] += delta * delta2


            # if debug: print('\tAdded',n_loop,'/',len(arrays[target]),'to follow_loops @',task_order[-1], target)
            if FULL_STARTS:
                # tsort = np.argsort(task_order)
                for ti, data_row in enumerate(task):
                    starts[tid, target, data_row:, task_order[ti]] = start[ti]
                    new_flags[tid, target, data_row, task_order[ti]] = flags[ti]
            elif PART_STARTS:
                new_flags[tid, target] = flags[-1]
                starts[tid, target] = start[-1]#  np.min(start)
                # if (absloc_count[tid, task_size, target] < 5):
                #     print('new flags', flags, new_flags)
                #     print('starts', start, starts)
                #     print(n_match2, partial_matches[tid][:min(5, n_match2)])
                # for i, j in zip(np.argsort(task_order), task_order):
                #     starts[tid, i] = start[j]

            # starts[tid, :, s_idx] = start

            # if debug: 
            #     # print('\tnew start for array',task_size-1,'from index',s_idx,'forward:', start[task_size-1])
            #     for ai in range(n_arrays):
            #         print('Array',ai,'can start at:')
            #         print(starts[tid, ..., ai],'- starts')
            #         print(new_flags[tid,...,ai].astype(np.int32), '- flags')
            # if first_pick:
            #     print()
            #     print(broke_early, n_match2)
            #     print(starts[tid])
            #     print(new_flags[tid])
            loop_count[tid] += n_loop
            if debug: 
                # n_loops += n_loop
                update[tid] += as_seconds_double(get_system_clock()-t)
                t = get_system_clock()
            if not broke_early and (n_match2 == 0): continue
            # print('first_pick',first_pick, 'n_match',n_match)

            if REORDERINGS and first_pick[tid]:
            #     # print('first selection')
                tasks_left += 1
                n_task2[tid] -= 1

                continue

            if (task_size+1) < n_arrays:
                if (tasks_left + n_match2) >= max_tasks[tid]:
                    additional = 2 * ((tasks_left + n_match2) + max_tasks[tid])
                    tasks_queue[tid] = np.append(tasks_queue[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
                    tasks_order[tid] = np.append(tasks_order[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
                    tasks_size[tid] = np.append(tasks_size[tid], np.empty((additional,), dtype=np.int32), axis=0)
                    max_tasks[tid] += additional

                # Shift existing tasks forward
                # if tid == 1:
                #     print('first 5 tasks:', tasks_order[tid][:min(tasks_left, 5)])
                #     print('sizes:', tasks_size[tid][:min(tasks_left, 5)])

                # # --- Store new tasks at beginning of array ---
                # curr_queue = tasks_queue[tid][tasks_left, :task_size].copy()
                # curr_order = tasks_order[tid][tasks_left, :task_size].copy()
                # # t = get_system_clock()
                # tasks_size[tid][n_match2:tasks_left+n_match2] = tasks_size[tid][:tasks_left].copy()
                # tasks_queue[tid][n_match2:tasks_left+n_match2] = tasks_queue[tid][:tasks_left].copy()
                # tasks_order[tid][n_match2:tasks_left+n_match2] = tasks_order[tid][:tasks_left].copy()
                # # copyt[tid] += get_system_clock()-t
                # tasks_size[tid][:n_match2] = task_size+1
                # tasks_queue[tid][:n_match2, :task_size] = curr_queue#tasks_queue[tid][tasks_left+n_match2, :task_size]
                # tasks_queue[tid][:n_match2, task_size] = partial_matches[tid, :n_match2][::-1]
                # tasks_order[tid][:n_match2, :task_size] = curr_order#tasks_queue[tid][tasks_left+n_match2, :task_size]
                # tasks_order[tid][:n_match2, task_size] = target

                # --- Store new tasks at the end of the array ---
                tasks_size[tid][tasks_left:tasks_left+n_match2] = task_size+1
                tasks_order[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_order[tid][tasks_left, :task_size]
                tasks_order[tid][tasks_left:tasks_left+n_match2, task_size] = target
                tasks_queue[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                tasks_queue[tid][tasks_left:tasks_left+n_match2, task_size] = partial_matches[tid, :n_match2][::-1]

                # tasks_size[tid][tasks_left:tasks_left+n_match2] = task_size+1
                # tasks_order[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_order[tid][tasks_left, :task_size]
                # tasks_order[tid][tasks_left:tasks_left+n_match2, task_size] = target
                # tasks_queue[tid][tasks_left:tasks_left+n_match2, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                # tasks_queue[tid][tasks_left:tasks_left+n_match2, task_size] = partial_matches[tid, :n_match2][::-1]


                # if tid == 1:
                #     print('Set tasks_order :',n_match2, ',',task_size,'=',target)
                #     for m in range(n_match2+tasks_left):
                #         print('\t\tTask',m,'-',tasks_order[tid][m,:task_size+1], tasks_size[tid][m])
                #     print('curr_queue',curr_queue)
                #     print('curr_order',curr_order)
                #     print('\tAdded',n_match2,'tasks: from',tasks_left,'to',tasks_left+n_match2,'target:',target)

                # if tasks_left == 0:
                #     print(tasks_queue[tid][:n_match2, :task_size+1])
                # if n_match2 > 0: print('added', n_match2, 'tasks with size', task_size+1, 'and', tasks_left,'tasks remaining')

                tasks_left += n_match2
                # if (tasks_left > 10000) and ((tasks_left % 10000) == 0):
                #     print(tasks_left, 'tasks_left')

            
            # Complete match of all arrays; write directly to inner_matches
            else:
                if (match_count + n_match2) >= max_matches[tid]:
                    additional = 2 * ((match_count + n_match2) + max_matches[tid])
                    inner_matches[tid] = np.append(inner_matches[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
                    max_matches[tid] += additional
                # if debug:
                #     print('\tFound',n_match2,'matches using partial task', tasks_queue[tid][tasks_left, :task_size])
                #     print('\t\tMatches:',partial_matches[tid, :n_match2])
                
                # inner_matches[tid][count:count+n_match2, :task_size] = tasks_queue[tid][tasks_left, :task_size]
                # inner_matches[tid][count:count+n_match2, task_size] = partial_matches[tid, :n_match2]
                inner_matches[tid][match_count:match_count+n_match2, task_order] = tasks_queue[tid][tasks_left, :task_size]
                inner_matches[tid][match_count:match_count+n_match2, target] = partial_matches[tid, :n_match2]
                match_count += n_match2
                n_match[tid] += n_match2
            if debug: lower[tid] += as_seconds_double(get_system_clock()-t)

            if allow_early_stop:
                if n_match.sum() >= num_samples:
                    tasks_left = 0
                    
            # if tid == CHECK_TID: print('task end', tid)
        if match_count:
            match = inner_matches[tid][:match_count]
            # n_match[tid] += match_count
            matches[tid].append(match.copy())

        # if debug: 
        #     print('\nFinished zero_index', zero_index)
        #     print('\tsetup',setup)
        #     print('\tupper',upper)
        #     print('\tmulti',multi)
        #     print('\tlower',lower)
        #     print('\tcount',match_count)
        #     print('\ttasks',n_task)
        #     print('\tloops',n_loops)
        #     print('\tset a',a_setter)
        #     # print(starts,'\n\n')

    if debug: 
        t = get_system_clock()

    if debug: 
        # print('\nFinished zero_index', zero_index)
        total = as_seconds_double(get_system_clock()-_start_t)
        print('\t\t\t\tsetup',setup.max(),'/',total)
        print('\t\t\t\tupper',upper.max(),'/',total)
        print('\t\t\t\tmulti',multi.max(),'/',total)
        print('\t\t\t\tupdate',update.max(),'/',total)
        print('\t\t\t\tlower',lower.max(),'/',total)
        # print('\tcount',match_count)
        # print('\t\t\t\ttasks',task_count.mean(),'/',total)
        # print('\t\t\t\tloops',loop_count.mean(),'/',total)
        print('\t\t\t\tset a',a_setter.max(),'/',total)

    # print('Total loops:', np.sum(loop_count)/1000000, 'million')
    # print('Total tasks:', np.sum(task_count)/1000000, 'million')
    # print('Total loops:', np.sum(loop_count))
    # print('Total tasks:', np.sum(task_count))
    # print('Timing:', np.sum(thomp))
    # if REORDERINGS:
    #     print(np.round(absloc_means[0], 2), '-means')
    #     # print(np.round(absloc_sqdif[0], 2), '-M2')
    #     # print(np.round(absloc_loops[0]/absloc_count[0], 2), '-mean2')
    #     print(absloc_count[0], '-count')
        # print(absloc_loops[0], '-score')
        # for tid in range(len(absloc_means)):
        #     vals = absloc_means[tid]#/absloc_count[tid]
        #     best = np.zeros((n_arrays,), dtype=np.int32)
        #     remaining = np.arange(1, n_arrays)

        #     for i in range(1, n_arrays):
        #         remain = np.ones((n_arrays,), dtype='bool')
        #         remain[best[:i]] = False
        #         remain = np.where(remain)[0] 
        #         best[i] = remain[np.argmin(vals[i, remain])]

        #     print('\tThread',tid,'best order:', best)

    # print('copy time:',as_seconds_double(np.sum(copyt)))
    result = np.empty((np.sum(n_match), n_arrays), dtype=np.int32)
    index  = np.int32(0)
    for i in range(n_threads):
        for match in matches[i]:
            result[index:index+len(match)] = match
            index += len(match)

    if debug: print('finalize:', as_seconds_double(get_system_clock()-t))
    # print('\t\t\t\tMulti:',as_seconds_double(np.mean(multi)),'/',as_seconds_double(get_system_clock()-_start_t))
    if shuffle: 
        np.random.shuffle(result)
    if allow_early_stop:
        result = result[:num_samples]
    return result


# USE_PARALLEL = False
# debug = False

# @nb.njit(
#     [i32[:,:](ListType(fx[:,::1]))#, ListType(fx[:,:,:]), i32[:])
#     for fx in INPUT_TYPES], 
# cache=True, nogil=True, parallel=USE_PARALLEL)
# def multiset_single_numba(arrays):#, resolutions, steps):

#     if debug: 
#         t = get_system_clock()
#         print('start')
#     n_arrays = len(arrays)
#     n_rows, n_cols = arrays[0].shape
#     n_features = n_cols // 3
#     dtype = arrays[0].dtype
#     finite_cols = np.empty((n_arrays, n_cols), dtype='bool')
#     # col_orders = np.empty((n_arrays, n_cols), dtype=np.int32)
#     # col_reorders = np.empty((n_arrays, n_cols), dtype=np.int32)
#     steps = np.empty((n_arrays, n_features+1), dtype=np.int32)

#     for i in range(n_arrays):
#         finite_cols[i] = get_finite_cols(arrays[i][:, :n_features])

#         # order_value = entropy[i]
#         # order_value[~finite_cols[i, :n_features]] = np.inf
#         # order = np.arange(len(order_value))
#         # order = np.argsort(order_value)[::-1]
#         # col_orders[i] = np.append(order, np.append(order + n_features, order+n_features*2))
#         # col_reorders[i] = np.argsort(col_orders[i])
#         # finite_cols[i] = finite_cols[i, col_orders[i]]
#         # arrays[i] = np.ascontiguousarray(arrays[i][:, col_orders[i]])
#         # steps[i] = np.append(np.cumsum(finite_cols[i][col_orders[i]]), 0) - 1
#         steps[i] = np.append(np.cumsum(finite_cols[i][:n_features]), 0) - 1

#     # array_skips = [create_skiplist_multi(a[:, s][:, r]) for a,s,r in zip(arrays, finite_cols, col_orders)]
#     array_skips = [create_skiplist_multi(a[:, s]) for a,s in zip(arrays, finite_cols)]

#     # array_skips = [create_skiplist_multi(a) for a in arrays]
#     # col_orders = col_orders[:, :n_features]
#     # Use _iget_num_threads to avoid caching issue with dynamic globals 
#     if USE_PARALLEL: n_threads = nb.np.ufunc.parallel._iget_num_threads()
#     else:            n_threads = 1
#     # check = np.zeros(n_threads, dtype='bool')
#     # for i in nb.prange(n_threads):
#     #     tid = nb.np.ufunc.parallel._iget_thread_id()
#     #     check[tid] = True
#     # n_threads = check.sum()


#     # print('n threads:', n_threads)
#     matches = List([List.empty_list(nb.int32[:,:]) for _ in range(n_threads)])
#     n_match = np.zeros((n_threads,), dtype=np.int32)
#     max_len = max([len(arr) for arr in arrays])
#     # index_array = np.zeros((n_threads, n_arrays), dtype=np.int32)
#     partial_matches = np.zeros((n_threads, max_len), dtype=np.int32)

#     max_matches = np.zeros(n_threads, dtype=np.int32) + 100
#     inner_matches = [np.empty((max_matches[0], n_arrays), dtype=np.int32) for _ in range(n_threads)]


#     # Keep (per thread) an array_location array which tracks at what index the 
#     # original arrays[] array is now located
#     #   This should allow adaptively swapping the order of arrays as a thread
#     #   processes more zero_indices (with the objective being to minimize n_tasks)

#     max_tasks = np.zeros(n_threads, dtype=np.int32) + 100
#     tasks = [np.asfortranarray(np.zeros((max_tasks[0], n_arrays), dtype=np.int32)) for _ in range(n_threads)]
#     task_array_index = [np.zeros((max_tasks[0],), dtype=np.uint16) for _ in range(n_threads)]

#     if debug: 
#         print('initialize:', as_seconds_double(get_system_clock()-t))
#         setup = 0
#         multi = 0
#         upper = 0
#         lower = 0

#     # print_every = int(max(0.05 * n_rows, 1))

#     for zero_index in nb.prange(n_rows):
#         if debug: t = get_system_clock()
#         # if (zero_index % print_every) == 0:
#         #     print(zero_index,'/',n_rows)

#         # Use _iget_thread_id to avoid caching issue with dynamic globals 
#         tid = nb.np.ufunc.parallel._iget_thread_id() if USE_PARALLEL else 0

#         # index_array = np.zeros((n_arrays,), dtype=np.int32)
#         # index_array[0] = zero_index
#         # task_array = index_array[tid]

#         # tasks = List([np.array([zero_index], dtype=np.int32)])
#         tasks[tid][0, 0] = zero_index
#         task_array_index[tid][0] = 1
#         task_count = 1

#         a = np.zeros((n_arrays, n_cols), dtype=dtype)


#         count = 0
#         n_task = 0
#         n_loops = 0
#         popped = 0
#         a_setter = 0
#         # part_reorder = np.empty((n_arrays, n_cols), dtype=np.int32)

#         if debug: setup += as_seconds_double(get_system_clock()-t)

#         # Change starts to per thread, so that they carry over for each zero_index       
#         starts = np.zeros((n_arrays,), dtype=np.int32)

#         while task_count > 0:
#             if debug: t = get_system_clock()

#             n_task += 1
#             if debug: t2 = get_system_clock()
#             task_count -= 1
#             array_index = task_array_index[tid][task_count]


#             if debug: popped += as_seconds_double(get_system_clock()-t2)


#             if debug: t2 = get_system_clock()
#             for i in range(array_index):
#                 a[i] = arrays[i][tasks[tid][task_count, i]]
#                 # part_reorder[i] = col_reorders[i]
#             if debug: a_setter += as_seconds_double(get_system_clock()-t2)
#             if debug: upper += as_seconds_double(get_system_clock()-t)


#             if debug: t = get_system_clock()
#             n_match2, n_loop = multiloop(
#                 partial_matches[tid], 
#                 a[:array_index], 
#                 arrays[array_index], finite_cols[:array_index] & finite_cols[array_index], 
#                 starts,#row_index, 
#                 array_skips[array_index], steps[array_index])#, col_orders[array_index])
#             n_loops += n_loop
#             if debug: multi += as_seconds_double(get_system_clock()-t)
#             # task_array[:array_index] = index_array[:array_index]
#             if debug: t = get_system_clock()

#             # extend tasks array when we reach current maximum, same as inner_matches
            
#             # pre-allocate partial tasks
#             if (array_index+1) < n_arrays:
#                 if (task_count + n_match2) >= max_tasks[tid]:
#                     additional = 2 * ((task_count + n_match2) - max_tasks[tid])
#                     tasks[tid] = np.append(tasks[tid], np.asfortranarray(np.empty((additional, n_arrays), dtype=np.int32)), axis=0)
#                     task_array_index[tid] = np.append(task_array_index[tid], np.empty((additional,), dtype=np.uint16), axis=0)
#                     max_tasks[tid] += additional

#                 task_array_index[tid][task_count:task_count+n_match2] = array_index+1
#                 tasks[tid][task_count:task_count+n_match2, :array_index] = tasks[tid][task_count, :array_index]
#                 tasks[tid][task_count:task_count+n_match2, array_index] = partial_matches[tid, :n_match2]
#                 task_count += n_match2

#             # write directly to inner_matches
#             else:
#                 if (count + n_match2) >= max_matches[tid]:
#                     additional = 2 * ((count + n_match2) - max_matches[tid])
#                     inner_matches[tid] = np.append(inner_matches[tid], np.empty((additional, n_arrays), dtype=np.int32), axis=0)
#                     max_matches[tid] += additional

#                 inner_matches[tid][count:count+n_match2, :array_index] = tasks[tid][task_count, :array_index]
#                 inner_matches[tid][count:count+n_match2, array_index] = partial_matches[tid, :n_match2]
#                 count += n_match2

#             if debug: lower += as_seconds_double(get_system_clock()-t)


#         if count:
#             match = inner_matches[tid][:count]
#             n_match[tid] += count
#             matches[tid].append(match)

#         if debug: 
#             print('\n', zero_index)
#             print('\tsetup',setup)
#             print('\tupper',upper)
#             print('\tmulti',multi)
#             print('\tlower',lower)
#             print('\tcount',count)
#             print('\ttasks',n_task)
#             print('\tloops',n_loops)
#             print('\t  pop',popped)
#             print('\tset a',a_setter)
#             print(starts)

#     if debug: 
#         print('multi', multi)
#         print('total', total)
#         t = get_system_clock()

#     result = np.empty((np.sum(n_match), n_arrays), dtype=np.int32)
#     index  = np.int32(0)
#     for i in range(n_threads):
#         for match in matches[i]:
#             result[index:index+len(match)] = match
#             index += len(match)

#     if debug: print('finalize:', as_seconds_double(get_system_clock()-t))
#     return result
