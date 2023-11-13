from itertools import starmap
import xarray as xr 
import numpy as np 
import pytest
import dask

from crest.crest.data.loading.Blockset import Blockset
from crest.crest.data.loading.Datafile import Datafile


example_coord = [
    {f'coord_{i}': c for i,c in enumerate(coord)} for coord in [
        # Non-uniform coordinate grid data
        [[1., 2., 4.], [0.1, 0.5, 0.6]],

        # Uniform coordinate grid data
        [[1., 2., 3.], [0.1, 0.2, 0.3]],
]]

example_data = [
    [10., 20., 30.],
    [40., 50., 60.],
    [70., 80., 90.],
]

create_xr = lambda coords, data=example_data: xr.Dataset(
    data_vars = {'var_0': (coords.keys(), data)},
    coords    = coords,
)

kws = {
    'features'     : ['var_0'],
    'extent'       : {'coord_0': [1, 4]},
    'window_depth' : {'coord_0': 0},
    'valid_percent': {'coord_0': 1},
}
datafile_default = [Datafile.load(ex)        for ex in map(create_xr, example_coord)]
datafile_defined = [Datafile.load(ex, **kws) for ex in map(create_xr, example_coord)]


@pytest.mark.parametrize('datafile', [datafile_default, datafile_defined])
@pytest.mark.parametrize('uniform',  [0, 1])
class TestDatafile:

    def test_dims(self, datafile: list[Datafile], uniform: int):
        assert(datafile[uniform].dims == ['coord_0', 'coord_1'])


    def test_valid_percent(self, datafile: list[Datafile], uniform: int):
        assert(datafile[uniform].valid_percent == {('coord_0',): 1, ('coord_1',): 1})


    def test_window_depth(self, datafile: list[Datafile], uniform: int):
        for dim in datafile[uniform].dims:
            assert((datafile[uniform].window_depth[dim] == np.array([0,0])).all())


    def test_resolution(self, datafile: list[Datafile], uniform: int):
        if uniform: assert(datafile[uniform].resolution == [1, 0.1])
        else:
            stack = lambda diff: np.c_[
                np.r_[diff[:1], diff],
                np.r_[diff, diff[-1:]],
            ]
            target = [stack(np.diff(c)) for c in example_coord[uniform].values()]
            output = datafile[uniform].resolution
            assert(len(output) == len(target))

            for out, tgt in zip(output, target):
                assert((out.round(5) == tgt.round(5)).all()), f'{out}\n{tgt}'


    def test_coord_array(self, datafile: list[Datafile], uniform: int):
        output = datafile[uniform].coord_array.compute().T
        target = np.array([
            np.array([example_coord[uniform]['coord_0']] * 3),
            np.array([example_coord[uniform]['coord_1']] * 3).T,
        ])
        assert((output == target).all()), f'{output}\n{target}\n'


    def test_calculate_overlap(self, datafile: list[Datafile], uniform: int):
        cases = [
            ({0: 1, 1: 1}, [1, 0.1], [False, False]), # Same resolution, not this datafile
            ({0: 0, 1: 0}, [1, 0.1], [True,   True]), # Same resolution, this datafile
            ({0: 5, 1: 5}, [10,  1], [False, False]), # 10*resolution
        ]
        for target, max_resolution, skip in cases:
            overlap = datafile[uniform].calculate_overlap(max_resolution, skip)
            assert(overlap == target)


    def test_update_blocks(self, datafile: list[Datafile], uniform: int):
        cases = [(1, 1, 1), (3, 3, 1), (3, 1, 1)]
        for block_num in cases:
            a = datafile[uniform].update_blocks(block_num)
            assert(datafile[uniform].numblocks == block_num)#, [a, datafile.numblocks, block_num]
    

    def test_update_chunks(self, datafile: list[Datafile], uniform: int):
        cases = [(1, 1, 1), (3, 3, 1), (3, 1, 1)]
        for chunksize in cases:
            a = datafile[uniform].update_chunks(chunksize)
            assert(datafile[uniform].chunksize == chunksize)#, [a, datafile.numblocks, block_num]


    def test_apply_overlap(self, datafile: list[Datafile], uniform: int):
        nan = np.nan 
        datafile[uniform].update_blocks((3,1,1))

        output = datafile[uniform].apply_overlap({0: 1, 1: 0})
        target = [
            [[[nan, nan, nan],
              [10., 20., 30.],
              [40., 50., 60.]]],

            [[[10., 20., 30.],
              [40., 50., 60.],
              [70., 80., 90.]]],
            
            [[[40., 50., 60.],
              [70., 80., 90.],
              [nan, nan, nan]]]
        ]

        for block, expect in zip(output, target, strict=True):
            if hasattr(block, '__call__'): block = block()
            data = np.rollaxis(block.data, 2)
            assert((np.nan_to_num(data) == np.nan_to_num(expect)).all())


    def test_apply_overlap_periodic(self, datafile: list[Datafile], uniform: int):
        nan = np.nan 
        datafile[uniform].update_blocks((1,3,1))

        output = datafile[uniform].apply_overlap({0: 0, 1: 1}, 'periodic')
        target = [
            [[[30., 10., 20.],
              [60., 40., 50.],
              [90., 70., 80.]]],

            [[[10., 20., 30.],
              [40., 50., 60.],
              [70., 80., 90.]]],
            
            [[[20., 30., 10.],
              [50., 60., 40.],
              [80., 90., 70.]]]
        ]

        for block, expect in zip(output, target, strict=True):
            if hasattr(block, '__call__'): block = block()
            print(block.data, block.data.shape)
            data = np.rollaxis(block.data, 2)
            print(data)
            assert((np.nan_to_num(data) == np.nan_to_num(expect)).all())


def test_verify_block_order():
    """ Ensure resolution vectors are matched up appropriately
        with the coordinates when blocks are created """ 
    random = np.random.default_rng(42).random
    series = lambda n: sorted((100*random(n)).round(5))
    shapes = (50, 20, 80)
    coords = {f'coord_{i}': series(n) for i,n in enumerate(shapes)}
    data   = random(np.prod(shapes)).reshape(shapes)

    datafile = Datafile.load(create_xr(coords, data))
    datafile.update_blocks((5,5,4))
    blocks   = datafile.apply_overlap({0: 3, 1: 1, 2: 5})
    blockset = Blockset(list(blocks))
    r_blocks = blockset._resolution 
    c_blocks = blockset._coords

    lens = [len(r_blocks), len(c_blocks)]
    assert(len(set(lens)) == 1), lens

    @dask.delayed
    def verify(r_block, c_block):
        for i, r in enumerate(r_block):
            bound = lambda j, v: slice(v[0], v[-1]+1) if i==j else v[0]
            valid = np.where( np.isfinite(c_block[..., i]) )
            index = tuple(starmap(bound, enumerate(valid))) + (i,)

            a = c_block[index]
            b = r[index[i]]
            e = np.diff(a).round(5) - b[:-1, 1]
            assert(len(a) == len(b)), [a, b, len(a), len(b)]
            assert(e < 1e-4).all(), f'Failure:\n{a}\n{b}\n{e}'

    for i in range(0, len(r_blocks), 50):
        chunk = slice(i, i+50)
        dask.compute(*map(verify, r_blocks[chunk], c_blocks[chunk]))


def test_invalid_location():
    with pytest.raises(FileNotFoundError):
        Datafile('not a valid location')

def test_invalid_extent():
    with pytest.raises(ValueError):
        Datafile(xr.Dataset(), extent={'coord_0': [1]})

def test_invalid_window_depth():
    with pytest.raises(ValueError):
        Datafile(xr.Dataset(), window_depth={'coord_0': [1,2,3]})

def test_invalid_valid_percent():
    with pytest.raises(ValueError):
        Datafile(xr.Dataset(), valid_percent={'coord_0': 1.1})


