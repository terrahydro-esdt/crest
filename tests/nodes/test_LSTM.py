import numpy as np
from crest.nodes import LSTM
import keras

def test_LSTM():
    """ test the CREST LSTM """
    features = 5
    units = 3
    batch_size = 2
    timesteps = 4

    keras.utils.set_random_seed(812)
    sm = LSTM('sm',inputs={'era5' : (features,), 'smap' :
                           (features,)},units=units)
    X =  {
            'era5' : np.random.uniform(size=[batch_size,timesteps,features]),
            'smap' : np.random.uniform(size=[batch_size,timesteps,features])
          }

    res = sm(X)

    keras.utils.set_random_seed(812)
    ans = keras.layers.LSTM(units=units)(keras.ops.concatenate([X['era5'],X['smap']],axis=-1))
    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),ans))))

    # Test save/load
    save_load = LSTM.decode(sm.encode())
    keras.utils.set_random_seed(812)
    res = save_load(X)
    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),ans))))

    # Test when passing an Input tensor
    _sm = LSTM('sm',inputs={'era5' : (features,), 'smap' :
                           (features,)},units=units)
    X =  {
            'era5' : keras.Input(shape=(timesteps,features)),
            'smap' : keras.Input(shape=(timesteps,features))
          }

    _sm(X)
