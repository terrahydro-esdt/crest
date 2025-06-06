from crest.nodes import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
from keras import Input

def test_LSTMCell():
    features = 5
    units = 3
    batch_size = 2

    set_random_seed(812)
    sm = LSTMCell('sm',inputs={'era5' : (features,), 'smap' :
                                   (features,)},units=units)

    h = tf.zeros([batch_size,units])
    c = tf.zeros([batch_size,units])

    X =  {
            'sm_h' : h,
            'sm_c' : c,
            'era5' : tf.random.uniform([batch_size,features]),
            'smap' : tf.random.uniform([batch_size,features])
          }

    res = sm(X)

    set_random_seed(812)
    _,[h,c] = tf.keras.layers.LSTMCell(units)(tf.concat([X['era5'],X['smap']],axis=-1),(h,c))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),[h,c]))))


def test_LSTM():
    features = 5
    units = 3
    batch_size = 2
    timesteps = 4

    set_random_seed(812)
    sm = LSTM('sm',inputs={'era5' : (features,), 'smap' :
                           (features,)},units=units)
    X =  {
            'era5' : tf.random.uniform([batch_size,timesteps,features]),
            'smap' : tf.random.uniform([batch_size,timesteps,features])
          }

    res = sm(X)

    set_random_seed(812)
    ans = tf.keras.layers.LSTM(units=units)(tf.concat([X['era5'],X['smap']],axis=-1))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),ans))))

    # Test when passing an Input tensor
    _sm = LSTM('sm',inputs={'era5' : (features,), 'smap' :
                           (features,)},units=units)
    X =  {
            'era5' : Input(shape=(timesteps,features)),
            'smap' : Input(shape=(timesteps,features))
          }

    _sm(X)
