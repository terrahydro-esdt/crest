import numpy as np
import keras
from crest.nodes import *

def test_CrossStitchLSTM():
    features = 2
    batch_size = 3
    timesteps = 4

    X =  {
            'era5' : np.random.uniform(size=[batch_size,timesteps,features]),
            'smap' : np.random.uniform(size=[batch_size,timesteps,features])
          }
    

    keras.utils.set_random_seed(812)
    clstm = CrossStitchLSTM(**{
        'name' : 'Kraken',
        'nodenames'  : ['sm','et'],
        'inputs' : {'era5' : (features,),'smap' : (features,)},
        'units' : [4,3]
        })
    res = clstm(X)

    sm_h = np.zeros([batch_size,4])
    sm_c = np.zeros([batch_size,4])
    et_h = np.zeros([batch_size,3])
    et_c = np.zeros([batch_size,3])

    keras.utils.set_random_seed(812)
    sm = keras.layers.LSTMCell(4)
    et = keras.layers.LSTMCell(3)

    inp = keras.ops.concatenate([X['era5'],X['smap']],axis=-1)
    for t in range(timesteps):
        x = keras.ops.squeeze(keras.ops.slice(inp,[0,t,0],[inp.shape[0],1,inp.shape[2]]),axis=1)

        _sm_h = sm_h # old smh
        y = keras.ops.concatenate([x,et_h],axis=-1)
        _,[sm_h,sm_c] = sm(y,(sm_h,sm_c))

        y = keras.ops.concatenate([x,_sm_h],axis=-1)
        _,[et_h,et_c] = et(y,(et_h,et_c))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['sm_h']),sm_h))))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['et_h']),et_h))))

    # Test save/load
    keras.utils.set_random_seed(812)
    save_load = CrossStitchLSTM.decode(clstm.encode())
    res = save_load(X)

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['sm_h']),sm_h))))

    assert(np.any(list(map(lambda x,y: x == y,
                          list(res['et_h']),et_h))))