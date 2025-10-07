import pytest
import shutil 
import os
import numpy as np
import xarray as xr
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Layer,LSTM,Lambda
from tensorflow.keras import Sequential
from tensorflow import TensorSpec,cast,stack,squeeze,concat
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.metrics import MeanSquaredError

from crest.data_server.DataServer import DataServer
from crest.model.Node import Node
from crest.model.Model import Model
from crest.data.loading import StructuredDataset,Dataset,Datafile
from crest.data import Batcher
from crest.base import BaseNode
from crest.data.transform import Transform


@pytest.mark.integtest
def test_soil_moisture_model():

    # Set seed for reproducibility
    seed = 812
    os.environ['PYTHONHASHSEED'] = str(812)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    set_random_seed(seed)

    # Copy data to local directory from data server
    ROOT_PATH = Path(DataServer.load('soil_moisture',os.getcwd()))

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

    # Define soil moisture model 
    class SoilMoistureModel(BaseNode):
        inputs = {
            'ERA5' : {k : {'datetime' : None, 'latitude' : 1, 'longitude' : 1} for k in features['ERA5']},
            'StaticAttributes.zarr/Irrigation' : {k : {'latitude' : 1, 'longitude' : 1} for k in features['Irrigation']},
            'StaticAttributes.zarr/Soil' : {k : {'latitude' : 1, 'longitude' : 1} for k in features['Soil']},
        }

        outputs = {'SMAP.zarr' : {'soil_moisture' : {'datetime' : 1, 'latitude' : 1, 'longitude' : 1}}}
                
        def __init__(self,stats):
            transforms = Transform(stats,by_feature={'*' : 'logexp'})
            super().__init__(transforms.standardize)
            self._temporal = LSTM(units=256, name='temporal')
            self._head = Dense(1, activation='relu')

        def call(self,X,training=False):
            era5 = stack(list(X['ERA5'].values()), axis=-1)
            era5 = squeeze(era5, axis=[2,3])
            x = self._temporal(era5)
            soil = stack(list(X['StaticAttributes.zarr/Soil'].values()), axis=-1)
            soil = squeeze(cast(soil, dtype=float), axis=[1,2])
            irrigation = stack(list(X['StaticAttributes.zarr/Irrigation'].values()), axis=-1)
            irrigation = squeeze(cast(irrigation, dtype=float), axis=[1,2])
            x = concat([x, soil, irrigation], axis=-1)
            return {'soil_moisture' : self._head(x)}

    # Training data
    extent_train = {'SMAP': {'extent' : {'datetime': ['2015-05-14 00:00:00', '2015-05-29 23:00:01']}},
                'ERA5': {'extent' : {}},
                'Soil': {'extent' : {}},
                'Irrigation': {'extent' : {}}
                }

    dataset_train = Dataset.from_models(
            verbose=False, **{
            'models'     : [SoilMoistureModel],
            'database_folder' : ROOT_PATH,
            'variable_depths' : {'datetime': (335, 0)},
            'datafile_kwargs' : extent_train 
            })

    batch_train = Batcher(dataset_train, **{
        'batch_size' : 10,
        'features'   : [INP , OUT],
        'repeat'     : True,
        'numblocks'  : [1,1,1],
        'workers'    : 0,
        'seed'       : 46
    })

    # Testing data / validation
    extent_test = {'SMAP': {'extent' : {'datetime': [np.datetime64('2015-05-30 00:00:00'), np.datetime64('2015-05-30 23:00:01')]}},
                    'ERA5': {'extent' : {}},
                    'Soil': {'extent' : {}},
                    'Irrigation': {'extent' : {}}
                    }

    dataset_test = Dataset.from_models(
            verbose=False, **{
            'models'     : [SoilMoistureModel],
            'database_folder' : ROOT_PATH,
            'variable_depths' : {'datetime': (335, 0)},
            'datafile_kwargs' : extent_test
            })

    batch_valid = Batcher(dataset_test, **{
        'batch_size' : 10,
        'features'   : [INP , OUT],
        'repeat'     : True,
        'duplicate'  : True,
        'numblocks'  : [1,1,1],
        'shuffle'    : False,
    })

    batch_test = Batcher(dataset_test, **{
        'batch_size' : 10,
        'features'   : [INP , OUT],
        'repeat'     : False,
        'duplicate'  : True,
        'numblocks'  : [1,1,1],
        'shuffle'    : False,
    })

        
    # Get stats
    stats = xr.concat(dataset_train.summary, dim='features')
    stats = stats.drop_duplicates('features', keep='last').compute()

    # Create HTG
    sm = SoilMoistureModel(stats)

    # Create model and fit
    model = Model(sm.graph)

    model.compile(**{
        'optimizer' : 'Adam', 
        'loss'      : [sm.losses],
        'metrics'      : [sm.loss]
    })

    with batch_train as train, batch_valid as valid:
        model.fit(
            train, 
            epochs=1, 
            validation_data=valid,
            steps_per_epoch=2, 
            validation_steps=9)

    # Evaluate
    with batch_test as test, batch_valid as valid:
        t = model.evaluate(test,steps=9,return_dict=True)
        v = model.evaluate(valid,steps=9,return_dict=True)
        assert t == v
    
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
        'features': [['x'],['y']],
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

    htg = Node(
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

    model.fit(bs, **{
        'batch_size': 5,
        'steps_per_epoch': 4,  # 20 samples / 5 samples per batch
        'epochs': 2,
        'verbose': False
    })

    bs.close()

    # Predition/evaluation kwargs
    pred_kwargs = {
        'batch_size': 5,
        'steps': 4,
        'verbose': False
    }

    # Create a Batcher for prediction/evaluation
    bp = Batcher(ds, **{
        'batch_size': 5,
        'features': ['x'],
        'repeat': False,
        'workers': 1,
        'duplicate': True,
        'shuffle': False
    })

    # Check predictions for different types of data
    using_bp = model.predict(bp)
    using_dict = model.predict({'x': x})
    assert np.allclose(using_dict['y'], using_bp['y'])
    
    bp.close()

    
    be = Batcher(ds, **{
        'batch_size': 5,
        'features': [['x'],['y']],
        'repeat': False,
        'workers': 1,
        'duplicate': True,
        'shuffle': False
    })

    # Check evaluations for different types of data
    using_be = model.evaluate(be, **pred_kwargs)
    using_dict = model.evaluate({'x': x},{'y': y},**pred_kwargs)
    assert np.allclose(using_dict, using_be)

    be.close()


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
        'features': (['x'],['y']),
        'repeat': False,
        'workers': 0,
        'shuffle': False,
        'duplicate': True
    })
    
    bp = Batcher(ds, **{
        'batch_size': 20,
        'features': (['x']),
        'repeat': False,
        'workers': 0,
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

    htg = Node(
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

    model.fit(bs,**{
        'steps_per_epoch': 5,  # 1000 samples / 20 samples per batch
        'epochs': 10,
        'verbose': False
    })
    bs.close()


    # Check predictions for different types of data
    using_bs = model.predict(bp)
 
    using_dict = model.predict({'x': x}, **{
        'batch_size' : 20
        })

    assert np.array_equal(using_dict['y'], using_bs['y'])
    bp.close()
