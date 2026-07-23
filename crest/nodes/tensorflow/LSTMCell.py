""" A node version of the Keras LSTMCell """

import dill
import tensorflow as tf
from ...model.Node import Node

class InitialState(tf.keras.Layer):
    """ Defines and returns the LSTMCell initial state set to all zeros. """
    def __init__(self,units):
        super().__init__()
        self.units = units

    def call(self,x):
        """ returns the initial state of all zeros """
        batch_size = tf.shape(x)[0]
        return tf.zeros([batch_size,self.units])


class LSTMCell(Node):
    """
    A Node implementing a Keras LSTMCell to be integrated
    within a HierarchalTensorGraph. The
    hidden and cell states have keys = "name_h" and "name_c".
    Multiple inputs will be concatenated (axis=-1) to form a single
    input to feed into the LSTMCell. LSTMCell states are initialized to zero.

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
        self.zeros = InitialState(self.units)

        # Add hiden and cell states to the inputs/outputs
        self.state_names = [name + '_' + i for i in ['h','c']]
        for i in self.state_names: inputs[i] = (units,)
        outputs = {i: (units,) for i in self.state_names}

        super().__init__(**{
            'node' : lambda X :  self.call(X),
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

        self.units = units
        self.kwargs = kwargs
        self._lstmcell = None

    # Set the lstm cell
    def build(self):
        """ build function creates the lstmcell """
        if not self._lstmcell:
            self._lstmcell = tf.keras.layers.LSTMCell(self.units,**self.kwargs)


    # Initialize states to zero
    def initial_state(self,x : dict) -> dict:
        """ LSTM initializatoin """
        inp = x[list(x.keys())[0]]
        s = [ self.zeros(inp) for i in self.state_names ]
        return dict(zip(self.state_names,s))

    def call(self,x) -> dict:
        """ Node callable """
        features = {k : v for k,v in x.items() if k not in self.state_names}
        inp = tf.keras.layers.Concatenate(axis=-1)(list(features.values()))
        [h,c] = [x[k] for k in self.state_names]
        _,[h,c] = self._lstmcell(inp,(h,c))

        return dict(zip(self.state_names,[h,c]))
