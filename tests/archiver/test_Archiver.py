import shutil
import random
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from crest.data.loading.Dataset import Dataset
from crest.data.loading.Datafile import Datafile
from crest.archiver.Archiver import Archiver


np.random.seed(1)
random.seed(1)

latitude = np.linspace(-90, 90, 90)
longitude = np.linspace(-180, 180, 90)
datetime = pd.date_range('2015-04-01 00:00:00',
                         '2015-08-31 23:00:00',
                         freq='H').values

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
_PREDICTIONS['latitude'] = np.repeat(
    np.random.choice(_DF2.data['latitude'], (3,)), 6)
_PREDICTIONS['longitude'] = np.repeat(
    np.random.choice(_DF2.data['longitude'], (3,)), 6)
_PREDICTIONS['datetime'] = np.repeat(
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
                         freq='H').values


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
                 data_schema=Dataset([_DF3, _DF4, _DF5]),
                 coords=coords)
    
    a = a.out_datafile
    
    latitude_out = _DF3.latitude[3:15]
    cond1 = np.all(a['latitude'] == latitude_out)
    
    longitude_out = _DF5.longitude[5:]
    cond2 = np.all(a['longitude'] == longitude_out)
    
    datetime_out = pd.date_range('2015-04-01 00:00:00',
                                 '2015-05-01 00:00:01',
                                 freq='H').values
    cond3 = np.all(np.array(a.datetime,
                            dtype='datetime64[ns]') == datetime_out)
    
    shutil.rmtree('tests/archiver/test.zarr')
    
    assert cond1 and cond2 and cond3
    
def test_create_schema_datafile():
    a = Archiver(output_path='tests/archiver/test9.zarr',
                 data_schema=_DF4,
                 coords=coords).out_datafile

    cond1 = np.all(a['latitude'] == _DF4.data['latitude'])
    cond2 = np.all(a['longitude'] == _DF4.data['longitude'])
    cond3 = np.all(a['datetime'] == _DF4.data['datetime'])

    shutil.rmtree('tests/archiver/test9.zarr')

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

    shutil.rmtree('tests/archiver/test10.zarr')

    assert cond1 and cond2 and cond3

def test_coord_match():

    a = Archiver(output_path='tests/archiver/test1.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 exact_coord_match=True,
                 coords=coords
                 )

    a = a._match_coords(pred_data=_PREDICTIONS)

    shutil.rmtree('tests/archiver/test1.zarr')

    cond_list = []
    for k in ['datetime', 'latitude', 'longitude']:
        cond_list.append(np.all(np.sort(a[k]) == np.sort(_PREDICTIONS[k])))

    assert np.sum(cond_list) == 3.

def test_aggregate():
    a = Archiver(output_path='tests/archiver/test2.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 exact_coord_match=True,
                 coords=coords
                 )

    a = a._aggregate(pred_df=pd.DataFrame.from_dict(_PREDICTIONS))
    a = a[['var2_pred']].reset_index().sort_values(
        ['latitude', 'longitude']).to_dict('list')
    shutil.rmtree('tests/archiver/test2.zarr')

    expected = {'datetime': [1.4283108e+18,
                             1.4283648e+18,
                             1.4299452e+18,
                             1.4284872e+18,
                             1.4287608e+18,
                             1.42884e+18,
                             1.428858e+18,
                             1.4292504e+18,
                             1.4296752e+18
                             ],
                'latitude': [35.3932584269663,
                             35.3932584269663,
                             35.3932584269663,
                             41.46067415730337,
                             41.46067415730337,
                             41.46067415730337,
                             47.52808988764045,
                             47.52808988764045,
                             47.52808988764045
                             ],
                'longitude': [-105.58659217877094,
                              -105.58659217877094,
                              -105.58659217877094,
                              -113.63128491620111,
                              -113.63128491620111,
                              -113.63128491620111,
                              -101.56424581005587,
                              -101.56424581005587,
                              -101.56424581005587
                              ],
                'var2_pred': [0.3365150537546911,
                              0.41891083825257325,
                              0.724187966763125,
                              0.30547667879131346,
                              0.5513266668054655,
                              0.8470835796730625,
                              0.7593640099528646,
                              0.8587585128800796,
                              0.44395168758584835
                              ]
                }

    cond_list = []
    for k in a.keys():
        cond_list.append(np.allclose(np.sort(a[k]), np.sort(expected[k])))

    assert np.sum(cond_list) == 4.

def test_archiver():

    with Archiver(output_path='tests/archiver/test3.zarr',
                  data_schema=Dataset([_DF1, _DF2]),
                  exact_coord_match=True,
                  coords=coords
                  ) as a:

        a.archive(_PREDICTIONS)

    output_data = xr.open_zarr('tests/archiver/test3.zarr')['var2_pred']
    written_value = float(output_data.mean(skipna=True))
    expected_value = a._aggregate(
        pd.DataFrame.from_dict(_PREDICTIONS))['var2_pred']
    expected_value = float(np.mean(expected_value))

    shutil.rmtree('tests/archiver/test3.zarr')

    assert written_value == expected_value

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

    _PREDICTIONS['lat_dummy'] = _PREDICTIONS.pop('latitude')
    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test8.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 coords=coords
                 ).archive(_PREDICTIONS)

    shutil.rmtree('tests/archiver/test8.zarr')
