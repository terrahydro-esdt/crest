from crest.model.HierarchalTensorGraph import HierarchalTensorGraph


class BaseNode(HierarchalTensorGraph):
    def __init__(self, node, name=None, inputs={}, outputs={}):
        super().__init__(node, name, inputs, outputs)
