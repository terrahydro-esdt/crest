import numpy as np 
import pytest

from crest.src.utils import find_neighbors 


def make2d(arr): 
    """ Ensure the passed value is at least 2 dimensional """
    arr = np.array(arr)
    return arr.reshape(-1, 1) if arr.ndim < 2 else arr 

def make_nested(arr):
    """ Create a ragged object array that contains (variable length) arrays """ 
    nested = np.empty(len(arr), dtype=object)
    nested[:] = list(map(np.array, arr))
    return nested 

def equal(output, expect):
    """ Check if the output and expected are equal """
    np_eq = lambda a, b: np.array_equal(np.array(a).flatten(), np.array(b).flatten())    
    inner = lambda a, b: (len(a)==len(b)) and all(map(np_eq, a, b))
    check = lambda a, b: (len(a)==len(b)) and all(map(inner, a, b))
    return check(output, expect)

def readable(*args, fmt='{} != {}'):
    """ Change numpy arrays into list form to make them readable """
    if len(args) > 1:             
        return fmt.format(*map(readable, args))

    if isinstance(args[0], (np.ndarray, tuple, list)):
        if isinstance(args[0][0], (np.ndarray, tuple, list)):
            return tuple(map(readable, args[0]))
        return list(map(readable, args[0]))
    return args[0]



def test_equal():
    # Double check numpy issue with empty arrays not affecting equal
    assert(equal([[make2d(0)]],  [[np.array(0)]]))
    assert(equal([[make2d([])]], [[np.array([])]]))
    assert(not equal([[make2d(0)]], [[np.array([])]]))


def test_1d():
    coords = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ]
    expect = [[2], [1], [0]]
    counts = [[1], [1], [1]]

    coords = list(map(make2d, coords))
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords)
    assert(equal(out, expect)), readable(out, expect)
    assert(equal(c,   counts)), readable(c,   counts)


def test_1d_resolutions():
    coords = [
        [1, 2, 3],
        [4, 5, 6],
        [3, 1, 0],
    ]
    expect = [[2], [0], [0]]
    counts = [[1], [1], [1]]

    coords = list(map(make2d, coords))
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords, resolutions=[2,2,2])
    assert(equal(out, expect)), readable(out, expect)
    assert(equal(c,   counts)), readable(c,   counts)
 

def test_1d_radius():
    coords = [
        [1, 2, 3],
        [4, 5, 6],
        [2, 1, 0],
    ]
    expect = [
        [[1], [2]],
        [[0], [0]],
        [[0], [0]],
    ]
    counts = [
        [1, 1],
        [1, 1],
        [1, 1],
    ]

    coords = list(map(make2d, coords))
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords, radius=2., p=np.inf)
    assert(equal(out, expect)), readable(out, expect)
    assert(equal(c,   counts)), readable(c,   counts)
 

# def test_1d_kwargs():
#     coords = [
#         [1, 2, 3],
#         [4, 5, 6],
#         [3, 1, 0],
#     ]
#     expect = [[2], [0], [0]]
#     counts = [[1], [1], [1]]

#     coords = list(map(make2d, coords))
#     expect = np.c_[list(map(make_nested, expect))]
#     counts = np.array(counts)
#     out, c = find_neighbors(coords, w=[0.25])
#     assert(equal(out, expect)), readable(out, expect)
#     assert(equal(c,   counts)), readable(c,   counts)
 

def test_2d():
    coords = [
        [[1,1], [1,2], [2,3]],
        [[2,1], [2,2], [3,3]],
        [[3,1], [3,2], [1,2]],
    ]
    expect = [
        [[1]],
        [[1]],
        [[2]]
    ]
    counts = [
        [1],
        [1],
        [1]
    ]

    coords = list(map(make2d, coords))
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords, radius=1)
    assert(equal(out, expect)), readable(out, expect)
    assert(equal(c,   counts)), readable(c,   counts)


def test_2d_chebyshev():
    coords = [
        [[1,1], [1,2], [2,3]],
        [[2,1], [2,2], [3,3]],
        [[3,1], [3,2], [1,3]],
    ]
    expect = [
        [[1], [2],    [2]],
        [[1], [1],    [2]],
        [[2], [1, 2], [1]],
    ]
    counts = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
    ]

    coords = list(map(make2d, coords))
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords, radius=1, p=np.inf, use_implode=True)
    assert(equal(out, expect)), readable(out, expect)   
    assert(equal(c,   counts)), readable(c,   counts)