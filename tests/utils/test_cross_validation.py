import random
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import pytest

from crest.data.loading.Datafile import Datafile
from crest.utils.cross_validation import KfoldSplit


np.random.seed(0)
random.seed(0)

latitude = np.linspace(-90, 90, 90)
longitude = np.linspace(-180, 180, 90)
datetime = pd.date_range('2015-04-01 00:00:00',
                         '2015-08-31 23:00:00',
                         freq='h').values

coords = ['datetime', 'latitude', 'longitude']

var1 = da.random.random((datetime.size, latitude.size, longitude.size),)

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

def test_get_second_fold():
    
    splitter = KfoldSplit(folds=3)
    
    df_train = Datafile(location=ds,
                        extent={'latitude': [24.396, 49.384],
                                'longitude': [-124.848, -66.885],
                                'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                            np.datetime64('2015-05-01 23:00:01')]
                                },
                        invalid_value=np.nan,
                        features=['var1'],
                        preprocessors=[splitter[2]['train']]
                        )
    
    df_test = Datafile(location=ds,
                        extent={'latitude': [24.396, 49.384],
                                'longitude': [-124.848, -66.885],
                                'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                            np.datetime64('2015-05-01 23:00:01')]
                                },
                        invalid_value=np.nan,
                        features=['var1'],
                        preprocessors=[splitter[2]['test']]
                        )
    
    train = df_train.data.to_dataset('features')['var1'].to_numpy()
    test = df_test.data.to_dataset('features')['var1'].to_numpy()
    combine_ds = np.nansum(np.stack((train,test)),0)
    
    assert np.all(np.isclose(combine_ds, 
                        ds.sel({'datetime': slice('2015-04-01','2015-05-01 23:00:01'),
                                'latitude': slice(24.396, 49.384),
                                'longitude': slice(-124.848, -66.885)})['var1'].to_numpy(),
                        equal_nan=True))
    
def test_get_all_folds():
    
    splitter = KfoldSplit(folds=3)
    
    df1 = Datafile(location=ds,
                    extent={'latitude': [24.396, 49.384],
                            'longitude': [-124.848, -66.885],
                            'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                        np.datetime64('2015-05-01 23:00:01')]
                            },
                    invalid_value=np.nan,
                    features=['var1'],
                    preprocessors=[splitter[1]['test']]
                    )
    
    df2 = Datafile(location=ds,
                    extent={'latitude': [24.396, 49.384],
                            'longitude': [-124.848, -66.885],
                            'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                        np.datetime64('2015-05-01 23:00:01')]
                            },
                    invalid_value=np.nan,
                    features=['var1'],
                    preprocessors=[splitter[2]['test']]
                    )
    
    df3 = Datafile(location=ds,
                    extent={'latitude': [24.396, 49.384],
                            'longitude': [-124.848, -66.885],
                            'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                        np.datetime64('2015-05-01 23:00:01')]
                            },
                    invalid_value=np.nan,
                    features=['var1'],
                    preprocessors=[splitter[3]['test']]
                    )
    
    
    test1 = df1.data.to_dataset('features')['var1'].to_numpy()
    test2 = df2.data.to_dataset('features')['var1'].to_numpy()
    test3 = df3.data.to_dataset('features')['var1'].to_numpy()
    combine_ds = np.nansum(np.stack((test1, test2, test3)),0)
    
    assert np.all(np.isclose(combine_ds,
                        ds.sel({'datetime': slice('2015-04-01','2015-05-01 23:00:01'),
                                'latitude': slice(24.396, 49.384),
                                'longitude': slice(-124.848, -66.885)})['var1'].to_numpy(),
                        equal_nan=True))
    
def test_select_axis_and_ordered():
        
        splitter = KfoldSplit(folds=3, split_type='ordered', axis=['datetime'])
    
        df_train = Datafile(location=ds,
                                extent={'latitude': [24.396, 49.384],
                                        'longitude': [-124.848, -66.885],
                                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                                np.datetime64('2015-05-01 23:00:01')]
                                        },
                                invalid_value=np.nan,
                                features=['var1'],
                                preprocessors=[splitter[2]['train']]
                                )
        
        df_test = Datafile(location=ds,
                                extent={'latitude': [24.396, 49.384],
                                        'longitude': [-124.848, -66.885],
                                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                                     np.datetime64('2015-05-01 23:00:01')]
                                        },
                                invalid_value=np.nan,
                                features=['var1'],
                                preprocessors=[splitter[2]['test']]
                                )
        
        train = df_train.data.to_dataset('features')['var1'].to_numpy()
        test = df_test.data.to_dataset('features')['var1'].to_numpy()
        combine_ds = np.nansum(np.stack((train,test)),0)
        
        assert np.all(np.isclose(combine_ds, 
                                ds.sel({'datetime': slice('2015-04-01','2015-05-01 23:00:01'),
                                        'latitude': slice(24.396, 49.384),
                                        'longitude': slice(-124.848, -66.885)})['var1'].to_numpy(),
                                equal_nan=True))

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_invalid_pixel_includes_unequal_folds():
        
        var2 = var1.copy()
        var2[:200,] = np.nan
        ds1 = xr.Dataset(
                        data_vars=dict(
                                var2=(["datetime", "latitude", "longitude"], var2)
                        ),
                        coords=dict(
                                datetime=datetime,
                                latitude=(["latitude"], latitude),
                                longitude=(["longitude"], longitude)
                        )
                        ).chunk('auto')
        
        splitter = KfoldSplit(folds=3,
                              split_type='ordered',
                              axis=['datetime'],
                              only_valid=False,
                              unequal_split={1:200},
                              )
        
        df_test = Datafile(location=ds1,
                                extent={'latitude': [24.396, 49.384],
                                        'longitude': [-124.848, -66.885],
                                        'datetime': [np.datetime64('2015-04-01 00:00:00'),
                                                     np.datetime64('2015-05-01 23:00:01')]
                                        },
                                invalid_value=np.nan,
                                features=['var2'],
                                preprocessors=[splitter[1]['test']]
                                )

        test = df_test.data.to_dataset('features')['var2'].to_numpy()
        
        assert np.nansum(test) == 0.
  
