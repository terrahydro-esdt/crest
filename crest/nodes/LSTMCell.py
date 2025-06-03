from crest.model.Node import Node
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K

class LSTMCell(Node):
    """
    A Node implementing a Keras LSTMCell to be integrated
    within an HierarchalTensorGraph recurrent node. The
    hidden and cell states have keys = "name_h" and "name_c".
    Multiple inputs will be concatenated (axis=-1) to form a single
    input to feed into the LSTMCell.
    
    Parameters
    ----------
    name    : str
        Name of the node
        
    inputs : dict
        The input keys and specs. All inputs should be shape=(batch_size,features).
        Multiple inputs will be concatenated (axis=-1) and fed into the LSTMCell.

    units : int
        The number of units in the LSTMCell

    **kwargs :
        The Keras LSTMCell kwargs
        
    """
    def __init__(self,
                 name : str,
                 inputs : dict,
                 units : int
                 ,**kwargs
                ):

        # Add hiden and cell states to the inputs/outputs
        self.state_names = [name + '_' + i for i in ['h','c']] 
        for i in self.state_names: inputs[i] = (units,)
        outputs = {i: (units,) for i in self.state_names}
            
        super().__init__(**{
            'node' : lambda X :  self._node(X),
            'name' : name,
            'inputs' : inputs,
            'outputs' : outputs
        })

             # Set as a recurrent node
        self.attributes['recurrent'] = True
        
        if('return_seq' in kwargs):
            self.attributes['return_seq'] = kwargs['return_seq']
            kwargs.pop('return_seq')

         # Initialize lstm cell
        self.lstm = keras.layers.LSTMCell(units,**kwargs)
        
    def _node(self,X): 
        # Concat features
        features = {k : v for k,v in X.items() if k not in self.state_names}
        inp = tf.concat(list(features.values()),axis=-1)
        batch_size = tuple(inp.shape)[0]
        # Get initial states or use recurrent states
        if(X[self.state_names[0]] is None):
            
            # If passing an inputTensor
            if(batch_size is None):
                h = tf.zeros(shape=(tf.shape(inp)[0],self.lstm.units))
                c = tf.zeros(shape=(tf.shape(inp)[0],self.lstm.units))
                _,[h,c] = self.lstm(inp,(h,c))
                return dict(zip(self.state_names,[h,c]))
            else:
                # Get batch size from inputs
                dtype = inp.dtype
                h,c = self.lstm.get_initial_state(batch_size=batch_size,dtype=dtype)
                _,[h,c] = self.lstm(inp,(h,c))
                return dict(zip(self.state_names,[h,c]))
        else:
            [h,c] = [X[k] for k in self.state_names]
        _,[h,c] = self.lstm(inp,(h,c))
        return dict(zip(self.state_names,[h,c]))