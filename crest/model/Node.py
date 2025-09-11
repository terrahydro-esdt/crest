from collections.abc import Callable
import dill
from .HierarchalTensorGraph import HierarchalTensorGraph as HTG
from .TensorSpec import TensorSpec
from .TensorGraph import ImproperTensorGraphError

class Node(HTG):
    """
    Node: Node is the object where individual ML models are defined. Nodes can then
    be used in conjunction with the HierarchalTensorGraph to build Hierarchal models
    composed of Nodes.

    Parameters
    ----------

    node : Callable
       The model.

    inputs : dict
        A dictionary with the name (keys) and tensor specifications (values) for the
        inputs to the model.

    outputs : dict
        A dictionary with the name (keys) and tensor specifications (values) for the
        outputs to the model.
    
    name : str, optional
       The name of the HTG which must be different than other nodes
       in the graph. If name is None, it will default to
       ['name', '__name__', '__qualname__'].

    **attr: dict
        The attributes of Node. Defaults:
        'recurrent' : False,
        'return_seq' : False,
        'initialization' : None,
        'roll_out' : None

    """
    def __init__(self,
                 node : Callable,
                 inputs: dict,
                 outputs: dict,
                 name: str | None = None,
                 **attr
                ):

        if isinstance(node,HTG):
            message = 'You cannot create a Node from a HierarchalTensorGraph.'
            raise ImproperTensorGraphError(message)

        super().__init__(name or HTG.get_name(node))

        self.node = node
        self.inputs = inputs
        self.outputs = outputs
        for k,v in attr.items():
            self.attributes[k] = v

        # Wrap tensor specs in CREST.TensorSpec
        for k in self.inputs:
            self.inputs[k] = TensorSpec(self.inputs[k])

        for k in self.outputs:
            self.outputs[k] = TensorSpec(self.outputs[k])

    @property
    def is_basenode(self) -> bool:
        """ Return basenode True """
        return True

    def encode(self,type='dill',**kwargs):
        keys = ['name','attributes','inputs','outputs','node']
        encode = {k:dill.dumps(v,**kwargs) for k,v in self.__dict__.items() if k in keys}
        return encode

    @classmethod
    def decode(cls,encode,type='dill',**kwargs):
        decode = {k:dill.loads(v,**kwargs) for k,v in encode.items()}
        return cls(**decode)
