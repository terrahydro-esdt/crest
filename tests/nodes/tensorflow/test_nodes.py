from crest.nodes.tensorflow import *
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

def test_CoupledLSTM():
    features = 2
    batch_size = 3
    timesteps = 4

    X =  {
            'era5' : tf.random.uniform([batch_size,timesteps,features]),
            'smap' : tf.random.uniform([batch_size,timesteps,features])
          }

    set_random_seed(812)
    clstm = CoupledLSTM(**{
        'name' : 'Kraken',
        'nodenames'  : ['sm','et'],
        'inputs' : {'era5' : (features,),'smap' : (features,)},
        'units' : [4,3]
        })

    res = clstm(X)

    sm_h = tf.zeros([batch_size,4])
    sm_c = tf.zeros([batch_size,4])
    et_h = tf.zeros([batch_size,3])
    et_c = tf.zeros([batch_size,3])

    set_random_seed(812)
    sm = tf.keras.layers.LSTMCell(4)
    et = tf.keras.layers.LSTMCell(3)

    inp = tf.concat([X['era5'],X['smap']],axis=-1)
    for t in range(timesteps):
        x = tf.squeeze(tf.slice(inp,begin =
                            [0,t,0],size=[inp.shape[0],1,inp.shape[2]]),axis=1)

        _sm_h = sm_h # old smh
        y = tf.concat([x,et_h],axis=-1)
        _,[sm_h,sm_c] = sm(y,(sm_h,sm_c))

        y = tf.concat([x,_sm_h],axis=-1)
        _,[et_h,et_c] = et(y,(et_h,et_c))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['sm_h']),sm_h))))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['et_h']),et_h))))
