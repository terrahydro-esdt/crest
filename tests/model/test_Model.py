import pytest
import shutil
import numpy as np
from pathlib import Path

from tensorflow.keras.layers import Dense,Dropout,Layer,LSTM,Lambda
from tensorflow.keras import Sequential
from tensorflow import TensorSpec,cast,stack,squeeze,concat
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.metrics import RootMeanSquaredError

from crest.data_server.DataServer import DataServer
from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
from crest.model.Model import Model
from crest.data.loading import StructuredDataset,Dataset,Datafile
from crest.data import Batcher

@pytest.mark.integtest
def test_soil_moisture_model():
    """ test a simple soil moisture model """ 
    # For reproducibility
    set_random_seed(812)
    
    # Copy data to local directory from data server
    ROOT_PATH = Path(DataServer.load('soil_moisture','./'))
    
    # Define data locations
    locations = {
        'ERA5'       : ROOT_PATH.joinpath('ERA5.zarr'),
        'SMAP'       : ROOT_PATH.joinpath('SMAP.zarr'),
        'Soil'       : ROOT_PATH.joinpath('StaticAttributes.zarr/Soil'),
        'Irrigation' : ROOT_PATH.joinpath('StaticAttributes.zarr/Irrigation'),
    
    }

    # Define data attributes
    features = {
        'ERA5'       : ['cape', 'cp', 'csfr','cvh', 'cvl', 'e', 'es',
                        'fal','lai_hv', 'lai_lv', 'msdwlwrf', 'msdwswrf',
                        'pev', 'sd','skt','smlt','sp', 'sro','slhf',
                        'sshf','ssro','stl1', 'stl2', 'stl3', 'stl4',
                        't2m','tp','u10', 'v10', 'z'],
        'Irrigation' : ['Actual', 'Equipped', 'GroundWater', 
                        'Non-Conventional', 'SurfaceWater'],
        'Soil'       : ['Clay_frac', 'Sand_frac', 'Silt_frac'],
        'SMAP'       : ['soil_moisture']
    }

    # Input and output features
    INP = features['ERA5'] + features['Soil'] + features['Irrigation']
    OUT = features['SMAP']

    # Define depth and temporal extents
    depth = {
        'ERA5': {'datetime': (335,0), 'latitude': 0, 'longitude': 0},
        'Irrigation': {'latitude': 0, 'longitude': 0},
        'Soil': {'latitude': 0, 'longitude': 0},
        'SMAP': {'datetime': 0, 'latitude': 0, 'longitude':0},
        }

    extent_train = {'SMAP':{'datetime': [np.datetime64('2015-05-14 00:00:00'), np.datetime64('2015-05-29 23:00:01')]},
                    'ERA5': {},
                    'Soil': {},
                    'Irrigation': {}
                    }

    extent_test = {'SMAP':{'datetime': [np.datetime64('2015-05-30 00:00:00'), np.datetime64('2015-05-30 23:00:01')]},
                    'ERA5': {},
                    'Soil': {},
                    'Irrigation': {}
                    }
    
    # Create Datasets and Batchers
    dataset_train = Dataset([
                    Datafile(
                        location     = locations[source],
                        features     = features[source],
                        window_depth = depth[source],
                        extent=extent_train[source],
                    ) for source in features])

    batch_train = Batcher(dataset_train, **{
        'batch_size' : 10,
        'features'   : [INP , OUT],
        'repeat'     : True,
        'numblocks'  : [1,1,1],
        'task_bytes' : 1e4,
        'workers'    : 0,
        'seed'       : 46
    })

    dataset_test = Dataset([
                    Datafile(
                        location     = locations[source],
                        features     = features[source],
                        window_depth = depth[source],
                        extent=extent_test[source],
                    ) for source in features])

    # Use validation set to test
    batch_valid = Batcher(dataset_test, **{
        'batch_size' : 10,
        'features'   : [INP , OUT],
        'repeat'     : True,
        'numblocks'  : [1,1,1],
        'task_bytes' : 1e4,
        'workers'    : 0,
        'seed'       : 46
    })

    batch_test = Batcher(dataset_test, **{
        'batch_size' : 10,
        'features'   : [INP , OUT],
        'repeat'     : False,
        'numblocks'  : [1,1,1],
        'task_bytes' : 1e4,
        'workers'    : 0,
        'shuffle'    : False
    })

    # Soil moisture model
    class sm_model(Layer):
        def __init__(self):
            super().__init__()
            self._temporal = LSTM(units=256, name='temporal')
            self._head = Dense(1, activation='relu')

        def __call__(self, X):

            era5 = stack([X[k] for k in features['ERA5']], axis=-1)
            era5 = Lambda(lambda x: squeeze(x, axis=[2,3]))(era5)
            x = self._temporal(era5)
            soil = stack([X[k] for k in features['Soil']], axis=-1)
            soil = squeeze(cast(soil, dtype=float), axis=[1,2])
            irrigation = stack([X[k] for k in features['Irrigation']], axis=-1)
            irrigation = squeeze(cast(irrigation, dtype=float), axis=[1,2])
            x = concat([x, soil, irrigation], axis=-1)

            output = {'soil_moisture': self._head(x)}

            return output
        
    # Create HTG basenode
    inputs = {f: TensorSpec((None,336, 1, 1)) 
              if f in features['ERA5'] else TensorSpec((None,1, 1)) for f in INP}

    htg = HierarchalTensorGraph(
        node=sm_model(),
        name='SM',
        inputs=inputs
    )
    
    # Create, compile, fit
    model = Model(htg)
    
    
    model.compile(**{
        'optimizer' : 'Adam', 
        'loss'      : 'mse', 
        'metrics'   : [RootMeanSquaredError()]
    })

    model.fit(
        batch_train, 
        validation_data=batch_valid, 
        epochs=3, 
        steps_per_epoch=7, 
        validation_steps=1)
    
    # Expected value
    ev = {
        'loss': 0.023858146741986275, 
        'root_mean_squared_error': 0.1544608324766159
    }
    
    # Evaluate
    res = model.evaluate(batch_test,return_dict=True)
    
    # Test
    assert res['loss'] == pytest.approx(ev['loss'], 1e-5)
 
    # Clean up
    shutil.rmtree(ROOT_PATH)
    
@pytest.mark.integtest
def test_model():
    """ Basic test of model interfaces """

    # Create some simple linear Dataset
    x = np.arange(0, 20, dtype=float)
    y = x
    ds = StructuredDataset(*[x, y], labels=['x', 'y'])

    # Create a Batcher for prediction/evaluation
    bs = Batcher(ds, **{
        'batch_size': 5,
        'features': ['x'],
        'repeat': True,
        'workers': 1,
        'duplicate': True,
        'shuffle': False
    })

    # Create some simple HTG
    layer = Sequential([
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(100),
        Dropout(0.1),
        Dense(1)
    ])

    htg = HierarchalTensorGraph(
        node=lambda X: {'y': layer(X['x'])},
        name='Dense',
        inputs={'x': TensorSpec(shape=[None, 1])},
        outputs={'y': TensorSpec(shape=[None, 1])}
    )

    # Build and fit the Model
    model = Model(htg)
    model.compile(**{
        'optimizer': 'Adam',
        'loss': 'mean_absolute_error'
    })

    model.fit(ds, **{
        'batch_size': 5,
        'steps_per_epoch': 4,  # 20 samples / 5 samples per batch
        'epochs': 20,
        'verbose': False
    })

    # Predition/evaluation kwargs
    pred_kwargs = {
        'batch_size': 5,
        'steps': 4,
        'verbose': False
    }

    # Check predictions for different types of data
    using_ds = model.predict(ds, **pred_kwargs)['y']
    using_bs = model.predict(bs, **pred_kwargs)['y']
    using_dict = model.predict({'x': x}, **pred_kwargs)['y']

    assert np.array_equal(using_ds, using_bs)
    assert np.array_equal(using_dict, using_bs)


    # Check evaluations for different types of data
    using_ds = model.evaluate(ds, **pred_kwargs)
    using_dict = model.evaluate({'x': x, 'y': y}, **pred_kwargs)
    assert (using_ds == using_dict)

    bs.close()

@pytest.mark.integtest
def test_model_exhaust():
    """ Basic test of model interfaces """

    # Create some simple linear Dataset
    x = np.arange(0, 1000, dtype=float)
    y = x
    ds = StructuredDataset(*[x, y], labels=['x', 'y'])

    # Create a Batcher for prediction/evaluation
    bs = Batcher(ds, **{
        'batch_size': 20,
        'features': ['x'],
        'repeat': False,
        'workers': 1,
        'shuffle': False,
        'duplicate': True
    })

    # Create some simple HTG
    layer = Sequential([
        Dense(10, activation='relu'),
        Dropout(0.3),
        Dense(10),
        Dropout(0.1),
        Dense(1)
    ])

    htg = HierarchalTensorGraph(
        node=lambda X: {'y': layer(X['x'])},
        name='Dense',
        inputs={'x': TensorSpec(shape=[None, 1])},
        outputs={'y': TensorSpec(shape=[None, 1])}
    )

    # Build and fit the Model
    model = Model(htg)
    model.compile(**{
        'optimizer': 'Adam',
        'loss': 'mean_absolute_error'
    })

    model.fit(ds, workers=1, **{
        'batch_size': 20,
        'steps_per_epoch': 5,  # 1000 samples / 20 samples per batch
        'epochs': 10,
        'verbose': False
    })

    # Predition/evaluation kwargs
    pred_kwargs = {
        'batch_size': 20,
        'steps': 5,
        'verbose': False,
    }

    kwards = pred_kwargs.copy()
    kwards['exhaust'] = True

    # Check predictions for different types of data
    using_ds = model.predict_exhaust(ds, **kwards)['y']
    using_bs = model.predict_exhaust(bs, **kwards)['y']
    using_dict = model.predict_exhaust({'x': x}, **kwards)['y']

    assert np.array_equal(using_ds, using_bs)
    assert np.array_equal(using_dict, using_bs)

    bs.close()

@pytest.mark.integtest
def test_deep_exhaust():
    """ Basic test of model interfaces """

    # Create some simple linear Dataset
    x = np.arange(0, 100, dtype=float)
    y = x
    ds = StructuredDataset(*[x, y], labels=['x', 'y'])

    # Create a Batcher for prediction/evaluation
    bs = Batcher(ds, **{
        'batch_size': 20,
        'features': ['x'],
        'repeat': False,
        'workers': 1,
        'shuffle': False,
        'duplicate': True
    })

    # Create some simple HTG
    layer = Sequential([
        Dense(10, activation='relu'),
        Dropout(0.3),
        Dense(10),
        Dropout(0.1),
        Dense(1)
    ])

    htg = HierarchalTensorGraph(
        node=lambda X: {'y': layer(X['x'])},
        name='Dense',
        inputs={'x': TensorSpec(shape=[None, 1])},
        outputs={'y': TensorSpec(shape=[None, 1])}
    )

    # Build and fit the Model
    model = Model(htg)
    model.compile(**{
        'optimizer': 'Adam',
        'loss': 'mean_absolute_error'
    })

    model.fit(ds, workers=1, **{
        'batch_size': 20,
        'steps_per_epoch': 5,  # 100 samples / 20 samples per batch
        'epochs': 10,
        'verbose': False
    })

    # Predition/evaluation kwargs
    pred_kwargs = {
        'batch_size': 20,
        'steps': 5,
        'verbose': False,
    }

    kwards = pred_kwargs.copy()
    kwards['exhaust'] = True

    # Check predictions for different types of data
    using_kr = model.model.predict(x, **pred_kwargs)['y']
    using_bs = model.predict_exhaust(bs, **kwards)['y']

    assert np.array_equal(using_kr, using_bs)

    bs.close()
