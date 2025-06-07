from .HierarchalTensorGraph import HierarchalTensorGraph as HTG
from .TensorSpec import TensorSpec
from collections.abc import Callable
from .TensorGraph import ImproperTensorGraphError

class Node(HTG):
    """
    Node: Node is the object where individual ML models are defined. Nodes can then
    be used in conjunction with the HierarchalTensorGraph to build Hierarchal models
    composed of Nodes.

    Parameters
    ----------

    name : str
       The name of the HTG which must be different than other nodes
       in the graph. If name is None, it will default to
       ['name', '__name__', '__qualname__'].

    node : Callable, optional
       The model.

    inputs : dict
        A dictionary with the name (keys) and tensor specifications (values) for the
        inputs to the model.

    outputs : dict
        A dictionary with the name (keys) and tensor specifications (values) for the
        outputs to the model.

    """
    def __init__(self,
                 node : Callable,
                 inputs: dict,
                 outputs: dict,
                 name: str | None = None,
                 **attr
                ):

        if(isinstance(node,HTG)):
            message = f'You cannot create a Node from a HierarchalTensorGraph.'
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

    def to_json():
        pass

    def from_json():
        pass



