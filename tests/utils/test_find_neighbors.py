from itertools import product
import pytest
import numpy as np 

from crest.utils import find_neighbors


def make2d(arr): 
    """ Ensure the passed value is at least 2 dimensional """
    arr = np.array(arr)
    return arr.reshape(-1, 1) if arr.ndim < 2 else arr 

def make_nested(arr):
    """ Create a ragged object array that contains (variable length) arrays """
    if len(arr) > 1 and isinstance(arr[0], (np.ndarray, tuple, list)):
        nested = np.empty(len(arr), dtype=object)
        nested[:] = list(map(np.array, arr))
        return nested 
    return np.atleast_1d(arr)

def explode(arr):
    """ Idempotent conversion to exploded representation (i.e. pd.explode) """
    expand = lambda pts: product(*map(np.atleast_1d, pts))
    return list(zip(*[pt for pts in zip(*arr) for pt in expand(pts)]))

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
        if len(args[0]) and isinstance(args[0][0], (np.ndarray, tuple, list)):
            return tuple(map(readable, args[0]))
        return list(map(readable, args[0]))
    return args[0]

def plot_2d_data(coords, expected, observed):
    """ Allow debugging 2d tests by plotting the grids and matched points """
    from collections import defaultdict as dd
    import matplotlib.pyplot as plt

    # Two plots: all matches; and missing/additional (difference) matches
    _, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Expand the imploded point representation
    expected = explode(expected)
    observed = explode(observed)

    # Use an empty list for each coordinate grid if no matches are given
    if not len(expected): expected = [[]] * len(coords)
    if not len(observed): observed = [[]] * len(coords)
    assert(len(coords) == len(expected) == len(observed))

    # Loop over each coordinate grid and the respective match indices
    match_exp = dd(list)
    match_obs = dd(list)
    pts_count = dd(int)
    for i, (grid, exp, obs) in enumerate(zip(coords, expected, observed)):
        print(f'\nGrid {i+1}:\n{grid}')
        print(f'\tObserved: {obs}')
        print(f'\tExpected: {exp}')

        # Plot the coordinate points, offset slightly when there are duplicates
        xs, ys = np.array(list(zip(*grid)), float)
        for j, pt in enumerate(map(tuple, grid)):
            xs[j] += pts_count[pt] / 20
            ys[j] += pts_count[pt] / 20
            pts_count[pt] += 1  
        axes[0].scatter(xs, ys)
        axes[1].scatter(xs, ys)

        # Gather the (x,y) locations for the match indices
        for pts, matches in [(exp, match_exp), (obs, match_obs)]:
            for j, pt in enumerate(pts):
                match = lambda v: np.atleast_1d(v[pt])[0]
                matches[j].append((match(xs), match(ys)))

    # Loop over each set of matches (expected points, and observed points)
    pts_count = dd(int)
    for i, matches in enumerate([match_exp, match_obs]):
        label, style = ('Result', '--') if i else ('Expect', '-')
        print(f'\n{label} has {len(matches)} matching points:')

        # Loop over matched point groups (one (x,y) per grid in each group)
        for group in matches.values():
            print(f'\t{group}')
            xs, ys = np.array(list(zip(*group)), float)

            # Plot a line between each point in the group, offset for repeats
            for j, m in enumerate(group):
                xs[j] += pts_count[m] / 100
                ys[j] += pts_count[m] / 100
                pts_count[m] += 1
            axes[0].plot(xs, ys, ls=style) 

    # Loop over matched point groups that are extra or missing from observed
    set_exp = set(map(tuple, match_exp.values()))
    set_obs = set(map(tuple, match_obs.values()))
    pts_count = dd(int)
    for i, differences in enumerate([set_exp-set_obs, set_obs-set_exp]):
        label, style = ('extra', '--') if i else ('missing', '-')
        print(f'\nThere are {len(differences)} {label} matches:')

        # Loop over matched point groups in the current set difference
        for group in differences:
            print(f'\t{group}')
            xs, ys = np.array(list(zip(*group)), float)

            # Plot a line between each point in the group, offset for repeats
            for j, m in enumerate(group):
                xs[j] += pts_count[m] / 100
                ys[j] += pts_count[m] / 100
                pts_count[m] += 1
            axes[1].plot(xs, ys, ls=style) 

    axes[1].set_yticklabels([])
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].set_title('Solid are Expected, Dashed are Observed')
    axes[1].set_title('Solid are Missing, Dashed are Extra')
    plt.tight_layout()
    plt.show()


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
    assert(c is None or equal(c, counts)), readable(c, counts)


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
    assert(c is None or equal(c, counts)), readable(c, counts)
 

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
    assert(c is None or equal(c, counts)), readable(c, counts)
 

""" kwargs not currently used """
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
        [[1,2], [3,1], [3,2]],
    ]
    expect = [
        [[1]],
        [[1]],
        [[0]]
    ]
    counts = [
        [1],
        [1],
        [1]
    ]

    coords = list(map(make2d, coords))
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords, radius=1, method='tree', p=2)
    assert(equal(out, expect)), readable(out, expect)
    assert(c is None or equal(c, counts)), readable(c, counts)


def test_2d_resolutions():
    grid1 = np.array([
        [1,1], [1,2], [1,3],
        [2,1], [2,2], [2,3],
    ])
    res1 = np.stack([[
        [1,1], [1,1], [1,1],
        [1,1], [1,1], [1,1], 
    ]]*2, axis=-1)
    grid2 = np.array([
        [0,2], [0,4],
        [1,2], [1,4],
    ])
    res2 = np.stack([[
        [1,2], [1,2],
        [1,2], [1,2], 
    ]]*2, axis=-1)
    grid3 = np.array([
        [0.5,0.5], [0.5,1.5],
        [1.0,0.5], [1.0,1.5],
        [1.5,0.5], [1.5,1.5], 
    ])
    res3 = np.stack([[
        [0.5,1.], [0.5,1.],
        [0.5,1.], [0.5,1.],
        [0.5,1.], [0.5,1.],
    ]]*2, axis=-1)

    coords = [grid1, grid2, grid3]
    resolu = [res1, res2, res3]
    expect = [[0,0,0,1,1,1],[2,2,2,2,2,2],[1,3,5,1,3,5]]

    expect = np.c_[list(map(make_nested, expect))]
    out, _ = find_neighbors(coords, resolu, radius=0.5)
    assert(equal(out, expect)), readable(out, expect)


def test_2d_small_radius():
    grid1 = np.array([
        [1,1], [1,2], [1,3],
        [2,1], [2,2], [2,3],
    ])
    res1 = np.stack([[
        [1,1], [1,1], [1,1],
        [1,1], [1,1], [1,1], 
    ]]*2, axis=-1)
    grid2 = np.array([
        [0,2], [0,4],
        [1,2], [1,4],
    ])
    res2 = np.stack([[
        [1,2], [1,2],
        [1,2], [1,2], 
    ]]*2, axis=-1)
    grid3 = np.array([
        [0.5,0.75], [0.5,1.75],
        [1.0,0.75], [1.0,1.75],
        [1.5,0.75], [1.5,1.75], 
    ])
    res3 = np.stack([[
        [0.5,1.], [0.5,1.],
        [0.5,1.], [0.5,1.],
        [0.5,1.], [0.5,1.],
    ]]*2, axis=-1)

    coords = [grid1, grid2, grid3]
    resolu = [res1, res2, res3]
    expect = [[1],[2],[3]]

    expect = np.c_[list(map(make_nested, expect))]
    out, _ = find_neighbors(coords, resolu, radius=0.25)
    assert(equal(out, expect)), readable(out, expect)


def test_2d_radius():
    coords = [
        [[1,1], [1,2], [2,3]],
        [[2,1], [2,2], [3,3]],
        [[1,2], [3,1], [3,2]],
    ]
    expect = [
        [[0, 1], [2],    [2]],
        [[0, 1], [1],    [2]],
        [[0],    [0, 2], [2]],
    ]

    coords = list(map(make2d, coords))
    expect = explode( np.c_[list(map(make_nested, expect))] )
    out, _ = find_neighbors(coords, radius=1)
    assert(equal(out, expect)), readable(out, expect)


def test_2d_chebyshev():
    coords = [
        [[1,1], [1,2], [2,3]],
        [[2,1], [2,2], [3,3]],
        [[1,3], [3,1], [3,2]],
    ]
    expect = [
        [[1], [2],    [2]],
        [[1], [1],    [2]],
        [[0], [0, 2], [2]],
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


def test_resolution_skips():
    """ 
    Verify duplicates along a coordinate dimension aren't skipped 
    when a non-uniform resolution might allow valid matches
    """

    # Test left side
    coords = list(map(make2d, [
        [[0, 0, 0]],
        [[1,-1,-1], [1,0,0]],
    ]))
    res = list(map(np.array, [
        [0, 0, 0],
        [ [[0,0], [0,0], [0,0]], [[1,0], [0,0], [0,0]] ],
    ])) # [ item0: [dim0: [left, right], dim1: ...], item1: ...]
    expect = np.array([[0], [1]])
    counts = np.array([[1], [1]])
    out, c = find_neighbors(coords, res, radius=1, p=np.inf)
    assert(equal(out, expect)), readable(out, expect)   
    assert(c is None or equal(c, counts)), readable(c, counts)

    # Test right side
    coords = list(map(make2d, [
        [[0, 0, 0]],
        [[-1,-1,-1], [-1,0,0]],
    ]))
    res = list(map(np.array, [
        [0, 0, 0],
        [ [[0,0], [0,0], [0,0]], [[0,1], [0,0], [0,0]] ],
    ])) # [ item0: [dim0: [left, right], dim1: ...], item1: ...]
    expect = np.array([[0], [1]])
    counts = np.array([[1], [1]])
    out, c = find_neighbors(coords, res, radius=1, p=np.inf)
    assert(equal(out, expect)), readable(out, expect)   
    assert(c is None or equal(c, counts)), readable(c, counts)


def test_non_uniform():
    coords = [
        [1,          9,     13, 15, 16],
        [1, 3, 5, 7, 9, 11, 13, 15],
    ]
    expect = [
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4],
        [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7],
    ]
    counts = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    coords = list(map(make2d, coords))
    resolu = [np.diff(c.T) for c in coords]
    expect = np.c_[list(map(make_nested, expect))]
    counts = np.array(counts)
    out, c = find_neighbors(coords, resolu)
    assert(equal(out, expect)), readable(out, expect)   
    assert(c is None or equal(c, counts)), readable(c, counts)


def test_empty():
    coords = [
        [1, 9, 13, 15, 16],
        [18, 19, 20, 21],
        [1,4,7,10],
        [-3,-2,-1,0,1,2],
    ]
    coords = list(map(make2d, coords))
    resolu = [np.diff(c.T) for c in coords]
    result = find_neighbors(coords, resolu)
    assert(sum(map(np.size, result)) == 0), result 
