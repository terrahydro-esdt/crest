import crest
from collections.abc import Callable
import copy

class RecurrentNode():
    def __init__(self, recurrence: int, node: None | Callable = None, name: None | str = None, inputs: dict = {}, outputs: dict = {}):
        """
        Initializes the RecurrentNode with the given recurrence, name, inputs, and outputs.
        """

        if (recurrence <= 0):
            raise Exception('Recurrence must be a greater than 0')
        
        self.parent = crest.model.HierarchalTensorGraph(name=name)

        node_names = [name + '_' + str(index) for index in range(recurrence)]

        # create duplicate nodes
        nodes = []
        for recurr_node in node_names:
            htg_copy = copy.deepcopy(node)
            htg_copy.name = recurr_node
            nodes.append(htg_copy)

        self.parent.add_edges_from([(nodes[i], nodes[i+1]) for i in range(recurrence - 1)])
        