from collections.abc import Callable
from ...model import Node

class TFNode(Node):
    """ Tensorflow node type """
    def __init__(self,
                 node : Callable,
                 inputs: dict,
                 outputs: dict,
                 name: str | None = None,
                 **attr
                ):

        super().__init__(node,inputs,outputs,name,**attr)
