import numpy as np
import tensorflow as tf
from crest.nodes.tensorflow import LSTM

def test_LSTM():
    """ test the CREST LSTM """
    features = 5
    units = 3
    batch_size = 2
    timesteps = 4

    tf.keras.utils.set_random_seed(812)
    sm = LSTM('sm',inputs={'era5' : (features,), 'smap' :
                           (features,)},units=units)
    X =  {
            'era5' : tf.random.uniform([batch_size,timesteps,features]),
            'smap' : tf.random.uniform([batch_size,timesteps,features])
          }

    res = sm(X)

    tf.keras.utils.set_random_seed(812)
    ans = tf.keras.layers.LSTM(units=units)(tf.concat([X['era5'],X['smap']],axis=-1))
    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),ans))))

    # Test save/load
    save_load = LSTM.decode(sm.encode())
    tf.keras.utils.set_random_seed(812)
    res = save_load(X)
    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),ans))))


    # Test when passing an Input tensor
    _sm = LSTM('sm',inputs={'era5' : (features,), 'smap' :
                           (features,)},units=units)
    X =  {
            'era5' : tf.keras.Input(shape=(timesteps,features)),
            'smap' : tf.keras.Input(shape=(timesteps,features))
          }

    _sm(X)
