import pytest
import xarray as xr 
import numpy as np 

from crest.src.data import Datafile


example_data = xr.Dataset(
    data_vars = {'var_1':
        (['coord_1', 'coord_2'], 
        [[10., 20., 30.],
         [40., 50., 60.],
         [70., 80., 90.]]) },
    coords = {
        'coord_1' : [1.,  2.,  3.],
        'coord_2' : [0.1, 0.2, 0.3],
    },
)

datafile_default = Datafile.load(example_data)
datafile_defined = Datafile.load(example_data, **{
    'features'     : ['var_1'],
    'extent'       : {'coord_1': [1, 3]},
    'window_depth' : {'coord_1': 0},
    'valid_percent': {'coord_1': 1},
})


@pytest.mark.parametrize('datafile', [datafile_default, datafile_defined])
class TestDatafile:

    def test_dims(self, datafile: Datafile):
        assert(datafile.dims == ['coord_1', 'coord_2'])


    def test_valid_percent(self, datafile: Datafile):
        assert(datafile.valid_percent == {('coord_1',): 1, ('coord_2',): 1})


    def test_window_depth(self, datafile: Datafile):
        for dim in datafile.dims:
            assert((datafile.window_depth[dim] == np.array([0,0])).all())


    def test_resolution(self, datafile: Datafile):
        assert(datafile.resolution == [1, 0.1])


    def test_coord_array(self, datafile: Datafile):
        expected = np.array([
         [[1.,  2.,  3. ],
          [1.,  2.,  3. ],
          [1.,  2.,  3. ]],

         [[0.1, 0.1, 0.1],
          [0.2, 0.2, 0.2],
          [0.3, 0.3, 0.3]]
        ])
        assert((datafile.coord_array.compute().T == expected).all())


    def test_calculate_overlap(self, datafile: Datafile):
        cases = [
            ({0: 1, 1: 1}, [1, 0.1], [False, False]), # Same resolution, not this datafile
            ({0: 0, 1: 0}, [1, 0.1], [True,   True]), # Same resolution, this datafile
            ({0: 5, 1: 5}, [10,  1], [False, False]), # 10*resolution
        ]
        for expected, max_resolution, skip in cases:
            overlap = datafile.calculate_overlap(max_resolution, skip)
            assert(overlap == expected)


    def test_update_chunks(self, datafile: Datafile):
        cases = [(1, 1, 1), (3, 3, 1), (3, 1, 1)]
        for block_num in cases:
            datafile.update_chunks(block_num)
            assert(datafile.numblocks == block_num)


    def test_apply_overlap(self, datafile: Datafile):
        nan = np.nan 
        datafile.update_chunks((3,1,1))

        blockset = datafile.apply_overlap({0: 1, 1: 0})
        expected = [
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
        assert(len(blockset) == len(expected))

        for block, target in zip(blockset, expected):
            data = np.rollaxis(block.data, 2)
            assert((np.nan_to_num(data) == np.nan_to_num(target)).all())


    def test_apply_overlap_periodic(self, datafile: Datafile):
        nan = np.nan 
        datafile.update_chunks((1,3,1))

        blockset = datafile.apply_overlap({0: 0, 1: 1}, 'periodic')
        expected = [
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
        assert(len(blockset) == len(expected))

        for block, target in zip(blockset, expected):
            data = np.rollaxis(block.data, 2)
            print(data)
            assert((np.nan_to_num(data) == np.nan_to_num(target)).all())


def test_invalid_location():
    with pytest.raises(FileNotFoundError):
        Datafile('not a valid location')

def test_invalid_extent():
    with pytest.raises(ValueError):
        Datafile(xr.Dataset(), extent={'coord_1': [1]})

def test_invalid_window_depth():
    with pytest.raises(ValueError):
        Datafile(xr.Dataset(), window_depth={'coord_1': [1,2,3]})

def test_invalid_valid_percent():
    with pytest.raises(ValueError):
        Datafile(xr.Dataset(), valid_percent={'coord_1': 1.1})


