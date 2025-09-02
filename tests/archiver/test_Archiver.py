import shutil
import random
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from crest.data.loading import Dataset
from crest.data.loading import Datafile
from crest.archiver import Archiver


np.random.seed(1)
random.seed(1)

latitude = np.linspace(-90, 90, 90)
longitude = np.linspace(-180, 180, 90)
datetime = pd.date_range('2015-04-01 00:00:00',
                         '2015-08-31 23:00:00',
                         freq='h').values

coords = ['datetime', 'latitude', 'longitude']

var1 = da.full(shape=(datetime.size, latitude.size, longitude.size),
               fill_value=np.nan
               )

ds = xr.Dataset(
    data_vars=dict(
        var1=(["datetime", "latitude", "longitude"], var1)
    ),
    coords=dict(
        datetime=datetime,
        latitude=(["latitude"], latitude),
        longitude=(["longitude"], longitude)
    )
).chunk('auto')

_DF1 = Datafile(location=ds,
                extent={'latitude': [24.396, 49.384],
                        'longitude': [-124.848, -66.885],
                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                     np.datetime64('2015-05-01 23:00:01')]
                        },
                invalid_value=np.nan,
                features=['var1']
                )


latitude = np.linspace(-90, 90, 90)
longitude = np.linspace(-180, 180, 180)


var2 = da.full(shape=(datetime.size, latitude.size, longitude.size),
               fill_value=np.nan
               )

ds = xr.Dataset(
    data_vars=dict(
        var2=(["datetime", "latitude", "longitude"], var2)
    ),
    coords=dict(
        datetime=datetime,
        latitude=(["latitude"], latitude),
        longitude=(["longitude"], longitude)
    )
).chunk('auto')

_DF2 = Datafile(location=ds,
                extent={'latitude': [24.396, 49.384],
                        'longitude': [-124.848, -66.885],
                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                     np.datetime64('2015-05-01 23:00:01')]
                        },
                invalid_value=np.nan,
                features=['var2']
                )

_PREDICTIONS = {}
_COORDINATES = {}
_COORDINATES['latitude'] = np.repeat(
    np.random.choice(_DF2.data['latitude'], (3,)), 6)
_COORDINATES['longitude'] = np.repeat(
    np.random.choice(_DF2.data['longitude'], (3,)), 6)
_COORDINATES['datetime'] = np.repeat(
    np.random.choice(_DF2.data['datetime'], (9,)), 2)
_PREDICTIONS['var2_pred'] = np.random.random((18,))


# synthetic datafiles for data schema testing

latitude = np.linspace(-90, 90, 360)
longitude = np.linspace(-180, 180, 80)
datetime = pd.date_range('2015-04-01 00:00:00',
                         '2015-08-31 23:00:00',
                         freq='D').values


var2 = da.full(shape=(datetime.size, latitude.size, longitude.size),
               fill_value=np.nan
               )

ds = xr.Dataset(
    data_vars=dict(
        var2=(["datetime", "latitude", "longitude"], var2)
    ),
    coords=dict(
        datetime=datetime,
        latitude=(["latitude"], latitude),
        longitude=(["longitude"], longitude)
    )
).chunk('auto')

_DF3 = Datafile(location=ds,
                extent={'latitude': [24.396, 49.384],
                        'longitude': [-124.848, -66.885],
                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                     np.datetime64('2015-05-01 23:00:01')]
                        },
                invalid_value=np.nan,
                features=['var2']
                )

latitude = np.linspace(-90, 90, 60)
longitude = np.linspace(-180, 180, 180)
datetime = pd.date_range('2015-04-01 00:00:00',
                         '2015-08-31 23:00:00',
                         freq='h').values


var2 = da.full(shape=(datetime.size, latitude.size, longitude.size),
               fill_value=np.nan
               )

ds = xr.Dataset(
    data_vars=dict(
        var2=(["datetime", "latitude", "longitude"], var2)
    ),
    coords=dict(
        datetime=datetime,
        latitude=(["latitude"], latitude),
        longitude=(["longitude"], longitude)
    )
).chunk('auto')

_DF4 = Datafile(location=ds,
                extent={'latitude': [24.396, 35.],
                        'longitude': [-124.848, -91.],
                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                     np.datetime64('2015-06-01 23:00:01')]
                        },
                invalid_value=np.nan,
                features=['var2']
                )

latitude = np.linspace(-90, 90, 90)
longitude = np.linspace(-180, 180, 360)


var2 = da.full(shape=(latitude.size, longitude.size),
               fill_value=np.nan
               )

ds = xr.Dataset(
    data_vars=dict(
        var2=(["latitude", "longitude"], var2)
    ),
    coords=dict(
        latitude=(["latitude"], latitude),
        longitude=(["longitude"], longitude)
    )
).chunk('auto')

_DF5 = Datafile(location=ds,
                extent={'latitude': [24.396, 35.],
                        'longitude': [-124.848, -91.]
                        },
                invalid_value=np.nan,
                features=['var2']
                )
    
def test_create_schema_dataset_no_index():
    
    a = Archiver(output_path='tests/archiver/test.zarr',
                 data_schema=Dataset([_DF3, _DF4, _DF5]),)
    
    a = a.out_datafile
    
    latitude_out = _DF3.latitude[3:15]
    cond1 = np.all(a['latitude'] == latitude_out)
    
    longitude_out = _DF5.longitude[5:]
    cond2 = np.all(a['longitude'] == longitude_out)
    
    datetime_out = pd.date_range('2015-04-01 00:00:00',
                                 '2015-05-01 00:00:01',
                                 freq='h').values
    cond3 = np.all(np.array(a.datetime,
                            dtype='datetime64[ns]') == datetime_out)
    
    assert cond1 and cond2 and cond3
    
def test_create_schema_datafile():
    a = Archiver(output_path='tests/archiver/test9.zarr',
                 data_schema=_DF4,
                 coords=coords).out_datafile

    cond1 = np.all(a['latitude'] == _DF4.data['latitude'])
    cond2 = np.all(a['longitude'] == _DF4.data['longitude'])
    cond3 = np.all(a['datetime'] == _DF4.data['datetime'])

    assert cond1 and cond2 and cond3
    
def test_create_data_schema_dict():
    a = Archiver('tests/archiver/test10.zarr',
                 coords = coords,
                 data_schema={'latitude': np.linspace(-90, 90, 360),
                              'longitude': np.linspace(-180, 180, 80),
                              'datetime': pd.date_range('2015-04-01 00:00:00',
                                                         '2015-08-31 23:00:00',
                                                         freq='D').values}
                 ).out_datafile
    
    cond1 = np.all(a['latitude'] == np.linspace(-90, 90, 360))
    cond2 = np.all(a['longitude'] == np.linspace(-180, 180, 80))
    cond3 = np.all(a['datetime'] == pd.date_range('2015-04-01 00:00:00',
                                                  '2015-08-31 23:00:00',
                                                  freq='D').values)

    assert cond1 and cond2 and cond3
    
def test_create_data_schema_string():
    a = Archiver(output_path='tests/archiver/test11.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 datafile_index=_DF1.name.split('_')[1],)
    assert a.out_datafile['latitude'][0] == _DF1.data['latitude'][0]
    
def test_create_data_schema_index_dict():
    a = Archiver(output_path='tests/archiver/test12.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 datafile_index={'datetime':_DF1.name.split('_')[1], 'latitude':_DF1.name.split('_')[1], 'longitude': 0},)
    assert a.out_datafile['latitude'][0] == _DF2.data['latitude'][0]
    assert a.out_datafile['longitude'][0] == _DF1.data['longitude'][0]
    assert a.out_datafile['datetime'][0] == _DF1.data['datetime'][0]

def test_archiver():
    a = Archiver(output_path='tests/archiver/test3.zarr',
                 data_schema=Dataset([_DF1, _DF2]),)

    a.archive(_COORDINATES, _PREDICTIONS)
    a.close()

    out = xr.open_zarr('tests/archiver/test3_stats.zarr/')['var2_pred']
    for i in range(0, len(_PREDICTIONS['var2_pred']), 2):
        out_temp = out.sel({'datetime': _COORDINATES['datetime'][i+1],
                            'latitude': _COORDINATES['latitude'][i+1],
                            'longitude': _COORDINATES['longitude'][i+1],}).values
        
        assert np.sum(_PREDICTIONS['var2_pred'][i:i+2]) == out_temp[0]
        assert out_temp[1] == 2.

    shutil.rmtree('tests/archiver/test3_stats.zarr')
    
    out = xr.open_zarr('tests/archiver/test3.zarr/')['var2_pred']
    for i in range(0, len(_PREDICTIONS['var2_pred']), 2):
        out_temp = out.sel({'datetime': _COORDINATES['datetime'][i+1],
                            'latitude': _COORDINATES['latitude'][i+1],
                            'longitude': _COORDINATES['longitude'][i+1],}).values
        
        assert np.mean(_PREDICTIONS['var2_pred'][i:i+2]) == out_temp
    
    shutil.rmtree('tests/archiver/test3.zarr')
    

def test_no_schema():

    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test5.zarr',
                 coords=coords)

def test_out_of_range_datafile():

    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test6.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 datafile_index=5,
                 coords=coords)

def test_mismatch_coordinate():

    _COORDINATES['lat_dummy'] = _COORDINATES.pop('latitude')
    with pytest.raises(ValueError):
        a = Archiver(output_path='tests/archiver/test8.zarr',
                     data_schema=Dataset([_DF1, _DF2]),)
        a.archive(_COORDINATES, _PREDICTIONS)
