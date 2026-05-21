from __future__ import annotations
import dill
from .LSTMCell import LSTMCell
from ...model import HierarchalTensorGraph
from ...model.TensorGraph import ImproperTensorGraphError

class CrossStitchLSTM(HierarchalTensorGraph):
    """
    Creates a cross-stich network of LSTMCells.

    Parameters
    ----------

    name : str
       The name of the coupled LSTM

    nodenames : str | list(str)
       The names of the LSTMCell nodes.

    inputs : dict
        The inputs dictionary defining the input tensor specs.

    units : int | list(ints)
        The units of each LSTM cell. If int then all cells
        have the same number of units.

    **kwargs:
        return_seq=False. Whether to return the last state or whole sequence.
        return_states = False. Whether to return both hidden and cell states.
        Anyone of the keras LSTMCell kwargs.
    
    """
    def __init__(self,
                 name : str,
                 nodenames : str | list[str],
                 inputs : dict,
                 units : int | list[int],
                 **attr
                ):

        super().__init__(name,**attr)

        self._inputs = inputs

        if isinstance(nodenames,str):
            self.nodenames = [nodenames]
        else:
            self.nodenames = nodenames

        if isinstance(units,int):
            self.units = [units for i in self.nodenames]
        else:
            self.units = units
            if len(units) != len(self.nodenames):
                message = 'len(units) not equal to len(nodenames)'
                raise ImproperTensorGraphError(message)

        self.state_inputs = {}
        for n,u in zip(self.nodenames,self.units):
            self.state_inputs[n] = {n + '_h' :  (u,)}

    def build(self):
        """ builds the graph """

        # Create lstmcells
        lstmcells = {}
        for n,u in zip(self.nodenames,self.units):
            # Add other node inputs
            _inputs = self._inputs.copy()
            for s,sv in self.state_inputs.items():
                if s != n:
                    for k,v in sv.items():
                        _inputs[k] = v

            # Create cell
            lstmcells[n] = LSTMCell(**{
                'name' : n,
                'inputs' : _inputs,
                'units'  : u
            })

        nodes = list(lstmcells.values())

        # Add I/O edges
        for i in nodes:
            self.add_edge('input',i,features=list(self._inputs.keys()))
            self.add_edge(i,'output')

        # Add self adges
        for i in nodes:
            self.add_edge(i,i)

        for i in lstmcells:
            for j in lstmcells:
                if i != j:
                    self.add_edge(i,j,features=[i +'_h'])

        # Remove cell states
        for i in self.nodenames:
            self.outputs.pop(i + '_c')