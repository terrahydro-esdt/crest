import numpy as np 
import pytest

from crest.src.utils import find_neighbors 


def make2d(arr): 
    """ Ensure the passed value is at least 2 dimensional """
    arr = np.array(arr)
    return arr.reshape(-1, 1) if arr.ndim < 2 else arr 

def equal(output, expect):
    """ Check if the output and expected are equal """
    np_eq = lambda a, b: np.array_equal(a.flatten(), b.flatten())    
    inner = lambda a, b: (len(a)==len(b)) and all(map(np_eq, a, b))
    check = lambda a, b: (len(a)==len(b)) and all(map(inner, a, b))
    return check(output, expect)

def readable(*args, fmt='{} != {}'):
    """ Change numpy arrays into list form to make them readable """
    if len(args) > 1:             return fmt.format(*map(readable, args))
    if isinstance(args[0], list): return        list(map(readable, args[0]))
    return tuple(v.ravel().tolist() for v in args[0])



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
    expect = [(2, 1, 0)]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords)
    assert(equal(output, expect)), readable(output, expect)


def test_1d_resolutions():
    coords = [
        [1, 2, 3],
        [4, 5, 6],
        [2, 1, 0],
    ]
    expect = [(2, 0, 0)]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords, resolutions=[2,2,2])
    assert(equal(output, expect)), readable(output, expect)


def test_1d_radius():
    coords = [
        [1, 2, 3],
        [4, 5, 6],
        [2, 1, 0],
    ]
    expect = [
        (1,     0, [0,1,2]),
        (2, [0,1],   [0,1]),
    ]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords, radius=2.)
    assert(equal(output, expect)), readable(output, expect)


def test_1d_empty():
    coords = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ]
    expect = [
        (0, [], []),
        (1,  0, []),
        (2,  1,  0),
    ]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords, allow_empty=True)
    assert(equal(output, expect)), readable(output, expect)


def test_1d_kwargs():
    coords = [
        [1, 2, 3],
        [4, 5, 6],
        [2, 1, 0],
    ]
    expect = [(2, 0, 0)]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords, w=[0.25])
    assert(equal(output, expect)), readable(output, expect)
 

def test_2d():
    coords = [
        [[1,1], [1,2], [2,3]],
        [[2,1], [2,2], [3,3]],
        [[3,1], [3,2], [1,3]],
    ]
    expect = [
        (1,     1, 2),
        (2, [1,2], 2),
    ]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords, radius=1)
    assert(equal(output, expect)), readable(output, expect)
 

def test_2d_chebyshev():
    coords = [
        [[1,1], [1,2], [2,3]],
        [[2,1], [2,2], [3,3]],
        [[3,1], [3,2], [1,3]],
    ]
    expect = [
        (1, [0,1],     2),
        (2, [1,2], [1,2]),
    ]

    coords = list(map(make2d, coords))
    expect = [tuple(map(make2d, e)) for e in expect]
    output = find_neighbors(coords, radius=1, p=np.inf)
    assert(equal(output, expect)), readable(output, expect)