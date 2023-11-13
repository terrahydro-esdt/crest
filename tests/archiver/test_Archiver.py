import shutil
import random
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from crest.crest.data.loading.Dataset import Dataset
from crest.crest.data.loading.Datafile import Datafile
from crest.crest.archiver.Archiver import Archiver


np.random.seed(1)
random.seed(1)

latitude = np.linspace(-90, 90, 90)
longitude = np.linspace(-180, 180, 90)
datetime = pd.date_range('2015-04-01 00:00:00',
                         '2015-08-31 23:00:00',
                         freq='H').values


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


def test_create_schema():

    a = Archiver(output_path='tests/archiver/test.zarr',
                 data_schema=Dataset([_DF1, _DF2]))

    a = xr.open_zarr('tests/archiver/test.zarr')
    cond1 = np.all(a['latitude'] == _DF2.data['latitude'])
    cond2 = np.all(a['longitude'] == _DF2.data['longitude'])
    cond3 = np.all(a['datetime'] == _DF2.data['datetime'])

    shutil.rmtree('tests/archiver/test.zarr')

    assert cond1 and cond2 and cond3


def test_coord_match():

    a = Archiver(output_path='tests/archiver/test1.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 exact_coord_match=True
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
                 exact_coord_match=True
                 )

    a = a._aggregate(pred_data=_PREDICTIONS)
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

    a = Archiver(output_path='tests/archiver/test3.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 exact_coord_match=True
                 )

    a.archive(_PREDICTIONS)
    output_data = xr.open_zarr('tests/archiver/test3.zarr')['var2_pred']

    written_value = float(output_data.mean(skipna=True))
    expected_value = a._aggregate(pred_data=_PREDICTIONS)['var2_pred']
    expected_value = float(np.mean(expected_value))

    shutil.rmtree('tests/archiver/test3.zarr')

    assert written_value == expected_value


def test_existing_data_no_vars():

    Archiver(output_path='tests/archiver/test4.zarr',
             data_schema=Dataset([_DF1, _DF2]))

    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test4.zarr')

    shutil.rmtree('tests/archiver/test4.zarr')


def test_no_schema():

    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test5.zarr')


def test_out_of_range_datafile():

    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test6.zarr',
                 data_schema=Dataset([_DF1, _DF2]),
                 datafile_index=5)


def test_invalid_file_suffix():

    with pytest.raises(NotImplementedError):
        Archiver(output_path='tests/archiver/test7.dummy',
                 data_schema=Dataset([_DF1, _DF2])
                 )


def test_mismatch_coordinate():

    _PREDICTIONS['lat_dummy'] = _PREDICTIONS.pop('latitude')
    with pytest.raises(ValueError):
        Archiver(output_path='tests/archiver/test8.zarr',
                 data_schema=Dataset([_DF1, _DF2])
                 ).archive(_PREDICTIONS)

    shutil.rmtree('tests/archiver/test8.zarr')
