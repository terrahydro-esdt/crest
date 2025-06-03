from .LSTMCell import LSTMCell
from crest.model.HierarchalTensorGraph import HierarchalTensorGraph

class LSTM(HierarchalTensorGraph):
    def __init__(self,name,inputs,units,**kwargs):
        super().__init__(name)
        self.lstmcell = LSTMCell(name,inputs,units,**kwargs)
        self.add_edge('input',self.lstmcell)
        self.add_edge(self.lstmcell,self.lstmcell)
        self.add_edge(self.lstmcell,'output')

        # Remove cell state by default
        if('return_state' not in kwargs):
            self.outputs.pop(name + '_c')
