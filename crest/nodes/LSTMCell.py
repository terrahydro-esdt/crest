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
    input to feed into the LSTMCell. States are initialized to zero.

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

        self.units = units

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

        # Check for return sequence and remove before passing to Keras
        if('return_seq' in kwargs):
            self.attributes['return_seq'] = kwargs['return_seq']
            kwargs.pop('return_seq')

        # Keras LSTMCell
        self.lstmcell = keras.layers.LSTMCell(units,**kwargs)

        # Set initialization
        self.attributes['initialization'] = self.initialize

    # Initialize states to zero
    def initialize(self,X : dict) -> dict:
        inp = X[list(X.keys())[0]]
        batch_size = tf.shape(inp)[0]
        s = [tf.zeros([batch_size,self.units]) for i in
             self.state_names]
        return dict(zip(self.state_names,s))

    def _node(self,X):

        # Concat features
        features = {k : v for k,v in X.items() if k not in self.state_names}
        inp = tf.concat(list(features.values()),axis=-1)
        [h,c] = [X[k] for k in self.state_names]
        _,[h,c] = self.lstmcell(inp,(h,c))

        return dict(zip(self.state_names,[h,c]))
