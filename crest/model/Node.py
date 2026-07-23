""" 
This module implements the foundation Node class of graphs.
All specialized Node classes should inherit from Node.
"""

from __future__ import annotations
from collections.abc import Callable
import dill
from .HierarchalTensorGraph import HierarchalTensorGraph as HTG
from .TensorSpec import TensorSpec
from .TensorGraph import ImproperTensorGraphError
from .IOSpec import IOSpec

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
        'roll_out' : None

    """
    def __init__(self,
                 node : Callable,
                 inputs: dict | IOSpec,
                 outputs: dict | IOSpec,
                 name: str | None = None,
                 **attr
                ):

        if isinstance(node,HTG):
            message = 'You cannot create a Node from a HierarchalTensorGraph.'
            raise ImproperTensorGraphError(message)

        super().__init__(name or HTG.get_name(node))

        self.node = node
        self._inputs = inputs.spec if isinstance(inputs,IOSpec) else inputs
        self._outputs = outputs.spec if isinstance(outputs,IOSpec) else outputs

        if isinstance(inputs,IOSpec):
            self._inputs_spec = inputs
        
        if isinstance(inputs,IOSpec):
            self._outputs_spec = outputs

        for k,v in attr.items():
            self.attributes[k] = v

        # Wrap tensor specs in CREST.TensorSpec
        for k in self._inputs:
            self.inputs[k] = TensorSpec(self.inputs[k])

        for k in self.outputs:
            self.outputs[k] = TensorSpec(self.outputs[k])

    @property
    def is_basenode(self) -> bool:
        """ This is how an HTG is recognized as a single node """
        return True
    
    @property
    def inputs(self):
        """ input specs """
        return self._inputs
    
    @property
    def outputs(self):
        """ output specs """
        return self._outputs

    def encode(self,type='dill',**kwargs):
        """ Default serialization dict """
        self.build()
        encode = {k:dill.dumps(v,**kwargs) for k,v in self.config.items()}
        return encode

    @classmethod
    def decode(cls,encode,type='dill',**kwargs):
        """ Default serialization decoder """
        decode = {k:dill.loads(v,**kwargs) for k,v in encode.items()}
        return cls(**decode)
