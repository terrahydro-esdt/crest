from crest.nodes.tensorflow import *
import numpy as np
import tensorflow as tf

def test_LSTMCell():
    features = 5
    units = 3
    batch_size = 2

    seed = 812
    tf.keras.utils.set_random_seed(seed)
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

    tf.keras.utils.set_random_seed(seed)
    _,[h,c] = tf.keras.layers.LSTMCell(units)(tf.concat([X['era5'],X['smap']],axis=-1),(h,c))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),[h,c]))))
   
    # Test save/load
    save_load = LSTMCell.decode(sm.encode())
    tf.keras.utils.set_random_seed(seed)
    res = save_load(X)
    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),[h,c]))))

