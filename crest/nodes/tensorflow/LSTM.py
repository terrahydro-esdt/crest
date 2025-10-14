import dill
from .LSTMCell import LSTMCell
from ...model import HierarchalTensorGraph

class LSTM(HierarchalTensorGraph):
    """ 
    
    LSTM HierarchalTensorGraph built using
    an LSTMCell recurrent Node. Note: it is more
    efficient to use a Keras LSTM layer wrapped in a node.
    
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

    def __init__(self,name,inputs,units,**kwargs):
        super().__init__(name)
        self.units = units
        self._inputs = inputs
        self.kwargs = kwargs

    def build(self):
        """ builds the graph """
        cell = LSTMCell(self.name + '_cell',self._inputs,self.units,**self.kwargs)
        self.add_edge('input',cell)
        self.add_edge(cell,cell)
        self.add_edge(cell,'output')

        # Remove cell state by default
        if 'return_state' not in self.kwargs:
            self.outputs.pop(self.name + '_cell' + '_c')
        else:
            if not self.kwargs['return_state']:
                self.outputs.pop(self.name + '_cell' + '_c')

    def encode(self,type='dill',**kwargs) -> dict:
        """ returns encoded dictionary """
        keys = ['_inputs','units','return_state','kwargs']
        encode = {k:dill.dumps(v) for k,v in self.__dict__.items() if k in keys}
        encode['inputs'] = encode.pop('_inputs')
        return {**encode,**super().encode()}
