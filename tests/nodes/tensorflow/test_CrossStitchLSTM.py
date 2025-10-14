import numpy as np
import tensorflow as tf
from keras import Input
from crest.nodes.tensorflow import *

def test_CrossStitchLSTM():
    features = 2
    batch_size = 3
    timesteps = 4

    X =  {
            'era5' : tf.random.uniform([batch_size,timesteps,features]),
            'smap' : tf.random.uniform([batch_size,timesteps,features])
          }

    tf.keras.utils.set_random_seed(812)
    clstm = CrossStitchLSTM(**{
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

    tf.keras.utils.set_random_seed(812)
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

    # Test save/load
    tf.keras.utils.set_random_seed(812)
    save_load = CrossStitchLSTM.decode(clstm.encode())
    res = save_load(X)

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['sm_h']),sm_h))))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['et_h']),et_h))))
