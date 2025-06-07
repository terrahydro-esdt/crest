import tensorflow as tf
from crest.model.Node import Node
from collections.abc import Callable

class BaseTFNode(Node):
    def __init__(self,
                 node : Callable,
                 inputs: dict,
                 outputs: dict,
                 name: str | None = None,
                 **attr
                ):

        super().__init__(node,inputs,outputs,name,**attr)

