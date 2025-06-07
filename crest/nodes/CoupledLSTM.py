from .LSTMCell import LSTMCell
from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
from crest.model.TensorGraph import ImproperTensorGraphError

class CoupledLSTM(HierarchalTensorGraph):
    """ 
    Creates a coupled LSTM.

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

    Returns
    -------
    
        

    """
    def __init__(self,
                 name : str,
                 nodenames : str | list[str],
                 inputs : dict,
                 units : int | list[int],
                 **kwargs
                ):
        
        super().__init__(name)
        
        if(isinstance(nodenames,str)):
            self.names = [nodenames]
        else:
            self.names = nodenames

        if(isinstance(units,int)):
            self.units = [units for i in self.names]
        else:
            self.units = units
            if(len(units) != len(self.names)):
                message = f'len(units) not equal to len(nodenames)'
                raise ImproperTensorGraphError(message)
                
        self.state_inputs = {}
        for n,u in zip(self.names,self.units):
            self.state_inputs[n] = {n + '_h' :  (u,)}
        
        # Create lstmcells
        self.lstmcells = {}
        for n,u in zip(self.names,self.units):

            # Add other node inputs
            _inputs = inputs.copy()
            for s,sv in self.state_inputs.items():
                if(s != n):
                    for k,v in sv.items():
                        _inputs[k] = v

            # Create cell
            self.lstmcells[n] = LSTMCell(**{
                'name' : n,
                'inputs' : _inputs,
                'units'  : u
            })

        nodes = list(self.lstmcells.values())

        # Add I/O edges
        for i in nodes:
            self.add_edge('input',i,features=list(inputs.keys()))
            self.add_edge(i,'output')

        # Add self adges
        for i in nodes:
            self.add_edge(i,i)
            
        for i in self.lstmcells:
            for j in self.lstmcells:
                if(i != j):
                    self.add_edge(i,j,features=[i +'_h'])

        # Remove cell states
        for i in self.names:
            self.outputs.pop(i + '_c')