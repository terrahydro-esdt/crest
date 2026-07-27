from crest.nodes import LSTMCell
import numpy as np
import keras

def test_LSTMCell():
    features = 5
    units = 3
    batch_size = 2

    seed = 812
    keras.utils.set_random_seed(seed)
    sm = LSTMCell('sm',inputs={'era5' : (features,), 'smap' :
                                   (features,)},units=units)
    
    h = np.zeros([batch_size,units])
    c = np.zeros([batch_size,units])

    X =  {
            'sm_h' : h,
            'sm_c' : c,
            'era5' : np.random.uniform(size=[batch_size,features]),
            'smap' : np.random.uniform(size=[batch_size,features])
         }

    res = sm(X)

    keras.utils.set_random_seed(seed)
    _,[h,c] = keras.layers.LSTMCell(units)(keras.ops.concatenate([X['era5'],X['smap']],axis=-1),(h,c))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),[h,c]))))
   
    # Test save/load
    save_load = LSTMCell.decode(sm.encode())
    keras.utils.set_random_seed(seed)
    res = save_load(X)
    assert(np.any(list(map(lambda x,y: x == y,
                          list(res.values()),[h,c]))))

