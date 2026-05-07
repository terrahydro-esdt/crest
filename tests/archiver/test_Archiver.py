from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import tempfile
import logging
import pytest
import tlz 

from crest.data.loading import Dataset
from crest.data.loading import Datafile
from crest.data.batching import Batcher
from crest.archiver import Archiver


def create_dataset():
    lat_1 = np.arange(70, 73, 0.1)
    lon_1 = np.arange(100, 103, 0.1)
    dt_1  = pd.date_range('2019-01-01', '2019-01-04', freq='D')
    dat_1 = np.arange(len(lat_1) * len(lon_1) * len(dt_1)).reshape((len(dt_1), len(lat_1), len(lon_1), 1))
    df_1  = Datafile(xr.DataArray(dat_1, {
        'datetime'  : dt_1,
        'latitude'  : lat_1,
        'longitude' : lon_1, 
        'features'  : ['f1'],
    }).to_dataset('features'))

    lat_2 = np.arange(70, 73, 0.35)
    lon_2 = np.arange(100, 103, 0.35)
    dt_2  = pd.date_range('2019-01-01', '2019-01-04', freq='h')
    dat_2 = np.arange(len(lat_2) * len(lon_2) * len(dt_2)).reshape((len(dt_2), len(lat_2), len(lon_2), 1))
    df_2  = Datafile(xr.DataArray(dat_2, {
        'datetime'  : dt_2,
        'latitude'  : lat_2,
        'longitude' : lon_2, 
        'features'  : ['f2'],
    }).to_dataset('features'))
    
    return Dataset([df_1, df_2])


def test_archiver():
    data = create_dataset()

    with tempfile.TemporaryDirectory() as temp:
        zarr = Path(temp).joinpath('test.zarr')
        original = data[0]._raw_data
        
        with Archiver(original, zarr) as archiver:
            with Batcher(data, batch_size=64, repeat=False, workers=1, log_level=logging.DEBUG) as b:
                for i, batch in enumerate(b.generator(show_timing=False)):
                    idx = tlz.merge_with(np.hstack, [tlz.dissoc(b.container[0]['coords'], 'features') for b in batch])
                    val = {'f1': np.stack([v.to_array(['f1']).flatten() for v in batch], axis=0).flatten()}
                    archiver.archive(idx, val)

        # Without mask_and_scale=False, int read as float due to NaN fill value
        written = xr.open_zarr(zarr, mask_and_scale=False)
        assert(list(written.coords) == list(original.coords))
        for d in original.coords:
            assert((written[d] == original[d]).all())

        for f in original:
            assert((written[f] == original[f]).all())
        assert(bool(xr.align(written, original, join='exact')))
                





                    