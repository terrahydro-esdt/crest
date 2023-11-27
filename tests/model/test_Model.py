import pytest
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow import TensorSpec
from crest import HierarchalTensorGraph
from crest import Model
import numpy as np
from crest.data.loading.StructuredDataset import StructuredDataset
from crest.data.Batcher import Batcher


# Temporary workaround for Heisenbug:
@pytest.mark.xfail
def test_Model():
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
    model.build(**{
        'optimizer': 'Adam',
        'loss': 'mean_absolute_error'
    })

    model.fit(ds, workers=1, **{
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

@pytest.mark.skip(reason="fails in CI/CD")
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
    model.build(**{
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

@pytest.mark.skip(reason="fails in CI/CD")
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
    model.build(**{
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
