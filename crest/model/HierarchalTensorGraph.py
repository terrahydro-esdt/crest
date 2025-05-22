from collections.abc import Callable
from functools import cache
from typing import Union
import importlib
import json
import pandas as pd
import copy
import matplotlib.colors as mcolors
import random as rand
import logging

# networkx breaks numpy if it's not also loaded, due to nx.lazy_import modifying sys modules
import networkx as nx
from .TensorSpec import TensorSpec

from .TensorGraph import TensorGraph, ImproperTensorGraphError
from .graphs.NetworkXGraph import NetworkXGraph

logger = logging.getLogger(__name__)


class HierarchalTensorGraph(TensorGraph):
    """
    HierarchalTensorGraph (HTG): The HTG is the core object where the Earth System Model (ESM)
    is encoded and specified. The HTG provides the information needed to build a CREST
    model using the Tensor Network backend. As the name suggest, HTG is a hierarchal graph object,
    and as such, nodes within a HTG are HTGs themselves.

    It has 2 modes:

    1.	A single node graph, referred to as a basenode, that represents a fundamental physical
    process in an ESM. The basenode is the main object intended for specifying an actual model
    of a physical process. As such, you must supply a callable function when instantiating a basenode.

    2.	A multi-node graph (nodes>1) used to represent ESM sub-systems. In this mode, HTG is not
    instantiated with callable function. Instead, nodes and edges are added to create a sub-system.
    Nodes can either be fundamental processes (basenode) or sub-systems (multi-node HTGs).


    Parameters
    ----------

    name : str
       The name of the HTG which must be different than other nodes
       in the graph. If name is None, it will default to
       ['name', '__name__', '__qualname__'].

    node : Callable, optional
       A callable function to initalize a HTG. If set, a
       single node (basenode) HTG is created.

    inputs : dict, optional
        A dictionary with the name (keys) and tensor specifications (values) for the
        input expected input tensors and produced output produced tensors. Note:
        these parameters are optional, however if not specified, in certain cases
        building the Model will fail. It's best practice to specify them!

    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __init__(self,
                 node: None | Callable = None,
                 name: None | str = None,
                 inputs: dict = {},
                 outputs: dict = {},
                 edges: list = []
                 ):

        self.name = name or HierarchalTensorGraph.get_name(node)
        self.node = node or self
        self.graph = NetworkXGraph()
        self.output = []
        self.inputs = inputs
        self.outputs = outputs
        self._inputs_map = {}  # dictionary specifying input feature renaming
        self._outputs_map = {}  # dictionary specifying input feature renaming
        self.is_recurrent = False

        # check that the HTG has a name
        if self.name == 'None':
            message = f'A HierarchalTensorGraph name must be given, but a {self.name} was found'
            raise ImproperTensorGraphError(message)

        # check that if node is passed it is callable
        if (self.node is not self) and (not callable(self.node)):
            message = f'node must be callable'
            raise ImproperTensorGraphError(message)

        # check that node is not an HTG
        if isinstance(node, HierarchalTensorGraph):
            message = "A HierarchalTensorGraph can not be used to"
            message += "create a basenode (i.e., node cannot be a"
            message += "HierarchalTensorGraph)"
            raise ImproperTensorGraphError(message)
        
        # log the creation of the HTG
        self.logger.info(f'Created HierarchalTensorGraph {self.name}')

        if (edges):
            self.add_edges_from(edges)

    def __iter__(self):
        """ Iterate through all nodes within the HTG """

        self.logger.info(f'Iterating through all nodes in HTG {self.name}')

        def traverse_nodes(graph, path):
            """ Recursive generator """
            for node in graph:
                yield [path + (graph.nodes[node]['htg'].name,), graph.nodes[node]['htg']]
                if not graph.nodes[node]['htg'].is_basenode:
                    yield from traverse_nodes(graph.nodes[node]['htg'].graph, path + (graph.nodes[node]['htg'].name,))

        yield from traverse_nodes(self.graph, ())

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def get_inputs(self, node):
        """ Get the inputs of the given node """

        self.logger.info(f'Getting inputs for node {node}')

        if node in self:
            input_maps = self[node]._inputs_map
            inputs = self[node].inputs

            inputs = [input_maps[x] if x in input_maps.keys()
                      else x for x in inputs]
            return inputs
        return None

    def get_outputs(self, node):
        """ Get the outputs of the given node """

        self.logger.info(f'Getting outputs for node {node}')

        if node in self:
            output_maps = self[node]._outputs_map
            outputs = self[node].outputs

            outputs = [output_maps[x] if x in output_maps.keys()
                       else x for x in outputs]
            return outputs
        return None

    def equals(self, node: Callable) -> bool:
        """ Check if the given node is the same as this node """

        self.logger.info(f'Checking if node {node} is equal to {self.node}')

        if self.node and self.is_basenode:
            if node and node.is_basenode:

                # compare names because it is impossible to compared callables
                return self.node.name == node.name
            else:
                return False

        if self.node.graph.is_isomorphic(node.graph.graph):
            return True

        return False

    def basenode_to_json(self):
        """
        Enables the serialization of the HTG basenode as a JSON string.
        """

        self.logger.info(f'Serializing basenode {self.name} to JSON')

        to_json = getattr(self.node, "to_json", None)

        if to_json:
            node_json = self.__dict__.copy()

            node_json.pop('graph')
            node_json.pop('output')

            if 'parent' in node_json:
                node_json.pop('parent')

            node_json['node'] = self.node.to_json()

            node_json['inputs'] = TensorSpec.dict_to_json(self.inputs)
            node_json['outputs'] = TensorSpec.dict_to_json(self.outputs)

            node_json['node_class'] = self.node.__class__.__name__
            node_json['node_module'] = self.node.__module__

            return json.dumps(node_json)
        else:
            raise Exception(
                'Base node <%s> does not have a to_json method' % self.name)

    def to_json(self):
        """
        Enables the serialization of the HTG network graph as a JSON string. 
        """

        self.logger.info(f'Serializing HTG {self.name} to JSON')

        graph_info = {}

        if self.is_basenode:
            if not type(self) == HierarchalTensorGraph:
                raise Exception(
                    'Base node <%s> is not of type HierarchalTensorGraph' % self.name)

            return self.basenode_to_json()

        # name
        graph_info['name'] = self.name

        # node
        graph_info['node'] = None

        # input and output map
        graph_info['inputs_map'] = self._inputs_map
        graph_info['outputs_map'] = self._outputs_map

        # input and output
        graph_info['inputs'] = TensorSpec.dict_to_json(self.inputs)
        graph_info['outputs'] = TensorSpec.dict_to_json(self.outputs)

        # edges
        graph_info['edges'] = list(self.edges)

        # nodes
        graph_info['nodes'] = []
        for name, node in self.nodes.items():
            if not name in ['input', 'output']:
                try:
                    graph_info['nodes'].append(node.to_json())
                except:
                    raise Exception(
                        f'Could not serialize node {name} in graph {self.name}')

        graph_info['node_class'] = self.node.__class__.__name__
        graph_info['node_module'] = self.node.__module__

        return json.dumps(graph_info)

    @staticmethod
    def basenode_from_json(graph_json: str):
        """

        Enables the deserialization of the HTG basenode represented as a JSON string.

        Parameters:
        -----------
        data : str
             The JSON string to deserialize.

        """

        graph_dict = graph_json
        if isinstance(graph_json, str):
            graph_dict = json.loads(graph_json)

        class_name = graph_dict['node_class']
        module_name = graph_dict['node_module']

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        from_json = getattr(class_, "from_json", None)

        if from_json:

            # the graph_json contains a sub dictionary to process
            if issubclass(class_, HierarchalTensorGraph):
                node = class_.from_json(graph_dict)
                return node

            htg = HierarchalTensorGraph(
                name=graph_dict['name'], node=class_.from_json(graph_dict['node']))

            htg.inputs = TensorSpec.json_to_dict(graph_dict['inputs'])
            htg.outputs = TensorSpec.json_to_dict(graph_dict['outputs'])

            htg._inputs_map = graph_dict['_inputs_map']
            htg._outputs_map = graph_dict['_outputs_map']

            return htg
        else:
            raise Exception(
                f'from_json was not defined for class {class_name}')

    @staticmethod
    def from_json(graph_json):
        """

        Enables the deserialization of the HTG network graph represented in a JSON structure. 

        Parameters:
        -----------
        data : dict
             The JSON string to deserialize.

        """

        graph_dict = json.loads(graph_json)

        # basenode
        if not graph_dict['node'] is None:
            return HierarchalTensorGraph.basenode_from_json(graph_dict)

        # graph
        graph = HierarchalTensorGraph(name=graph_dict['name'])

        # input and output
        graph.inputs = TensorSpec.json_to_dict(graph_dict['inputs'])
        graph.outputs = TensorSpec.json_to_dict(graph_dict['outputs'])

        # nodes
        for node_json in graph_dict['nodes']:
            graph.add_node(HierarchalTensorGraph.from_json(node_json))

        # edges
        graph.add_edges_from(graph_dict['edges'])

        # input and output map
        graph._inputs_map = graph_dict['inputs_map']
        graph._outputs_map = graph_dict['outputs_map']

        return graph

    @staticmethod
    def get_name(obj) -> str:
        """

        Try to determine the name of the given object
        using in order 'name', '__name__', and '__qualname__'.

        Parameters
        ----------
        obj : Callable, HierarchalTensorGraph
             The object from which to get the name

        """

        value = getattr(obj, 'obj', None)
        if value is None:
            for attr in ['name', '__name__', '__qualname__']:
                value = value or getattr(obj, attr, None)
            return value or str(obj)
        return HierarchalTensorGraph.get_name(value)

    def rename_io(self,
                  inputs_map: None | dict = None,
                  outputs_map: None | dict = None,
                  node: None | str | tuple | TensorGraph = None,
                  ):
        """
            Sets any renaming of input/output tensors needed.

            Parameters
            ----------
            node : specifies which node to apply the renaming. If None,
                    it applies it to itself.

            inputs_map : dict a dictionary of key-value pairs = (from name, to name).
            If str is used for keys, it will attempt a search for a unique match.
            If a tuple is used for keys, it will look for an exact match.

            outputs_map : dict a dictionary of key-value pairs = (from name, to name).
            If str is used for keys, it will attempt a search for a unique match.
            If a tuple is used for keys, it will look for an exact match.


        """

        self.logger.info(f'Renaming inputs and outputs for node {node}')
        
        if inputs_map:
            if not isinstance(inputs_map, dict):
                raise ImproperTensorGraphError('inputs_map must be a dict')
            self[node]._inputs_map = inputs_map

        if outputs_map:
            if not isinstance(outputs_map, dict):
                raise ImproperTensorGraphError('outputs_map must be a dict')
            self[node]._outputs_map = outputs_map

    def search(self, name: str, partial: bool = False) -> dict:
        """

        Tries to find the node with the given name.
        Returns match/matches as dict with key = tuple and
        value = node.

        Parameters
        ----------
        name : string or tuple
             The name serached for in the path

        partial : bool
             Whether to search for an exact or partial match. If tuple is given,
             only exact match will be used and partial will be ignored.


        """

        self.logger.info(f'Searching for node {name} in HTG {self.name}')

        # tuples must be exact matches.
        if isinstance(name, tuple):
            return dict(filter(lambda t: name == t[0], self.all_nodes.items()))

        if partial:
            return dict(filter(lambda s: name in ''.join(s[0]), self.all_nodes.items()))
        return dict(filter(lambda s: name in s[0], self.all_nodes.items()))

    @property
    def all_nodes(self) -> dict:
        """ Returns a dict of all child nodes within the parent """

        self.logger.info(f'Getting all nodes in HTG {self.name}')

        return dict([i for i in self])

    @property
    def edges(self):

        self.logger.info(f'Getting all edges in HTG {self.name}')

        return self.graph.edges

    @property
    def nodes(self) -> dict:
        """ Returns a dict of nodes within the parent HTG """

        self.logger.info(f'Getting all nodes in HTG {self.name}')

        return self.graph.get_node_attributes('htg')

    @property
    def sources(self) -> list:
        """ Nodes with no incoming edges """

        self.logger.info(f'Getting all sources in HTG {self.name}')

        def is_source(node): return self.graph.in_degree(node) == 0

        return list(filter(is_source, self.graph.nodes))

    @property
    def sinks(self) -> list:
        """ Nodes with no outgoing edges """

        self.logger.info(f'Getting all sinks in HTG {self.name}')

        def is_sink(node): return self.graph.out_degree(node) == 0

        return list(filter(is_sink, self.graph.nodes))

    @property
    def is_empty(self) -> bool:
        """ Check if graph is empty """

        self.logger.info(f'Checking if HTG {self.name} is empty')

        return self.graph.is_empty

    @property
    def is_basenode(self) -> bool:
        """ Check if graph is empty """

        self.logger.info(f'Checking if HTG {self.name} is a basenode')

        return self.graph.is_empty

    @staticmethod
    def identity(name: str) -> 'HierarchalTensorGraph':
        """
        Create an the identity HTG.

        name : string
             The name of the HTG

        """
        return HierarchalTensorGraph(lambda x: x, name)

    # TODO: Type hinting of HierarchicalTensorGraph is not working

    def get_node(self, node: Union[str, Callable, TensorGraph]):
        """Wrap the given node in a HTG and add to its graph if necessary.

        Parameters
        ----------
        node : str, callable, or HTG
            If a callable is passed, it is wrapped into a HTG and added to
            the graph if it doesn't yet exist. If a string is passed, the HTG
            (which must already exist in the graph) is returned. If a HTG is passed,
            it is added to the graph if it doesn't yet exist.

        Returns
        -------
        HierarchalTensorGraph
            HTG which represents the given node in the graph.

        Raises
        ------
        ImproperTensorGraphError
            - If a node with the same name already exists in the graph, but is
            not the same HTG object.
            - If a the same referenced HTG passed already exists in the HTG.
            - If node is not callable, a string, or a HierarchalTensorGraph

        """

        self.logger.info(f'Getting node {node} in HTG {self.name}')

        if not (isinstance(node, str) or isinstance(node, HierarchalTensorGraph) or callable(node)):
            message = f'node must be either callable, a string, or a HierarchalTensorGraph'
            raise ImproperTensorGraphError(message)

        # fetch node by its name
        if isinstance(node, str):
            # add io node if first time called
            if (node in ['input', 'output']) and (node not in self):
                io = HierarchalTensorGraph.identity(node)
                io.parent = self
                self.graph.add_node(io.name, htg=io)
                return io

            # otherwise if using a str() it must already be in the graph
            assert (node in self), f'Unknown name "{node}"'
            return self[node]

        # if not a name (str) and not of type(HierarchalTensorGraph)
        if not isinstance(node, HierarchalTensorGraph):

            # check if basenode with this name exist and if it has the same node value.
            # if it has the same node value then this refers to that basenode.
            name = HierarchalTensorGraph.get_name(node)
            if (name in self.graph) and (node is self[name].node):
                return self[name]

            # if it doesn't exist we need to create a basenode
            node = HierarchalTensorGraph(node)

        # check if you passed a different HTG with the same name
        if (node.name in self.graph) and (node is not self.nodes[node.name]):
            message = f'node ({node.name} = {node}) has the same name'
            message += f'(and path) as existing node {self.nodes[node.name]}'
            raise ImproperTensorGraphError(message)

        # if node is not in the graph add it
        if not node.name in self.graph:

            # nodes before adding
            before = self.all_nodes

            node.parent = self

            # add node to graph
            self.graph.add_node(node.name, htg=node)

            # check we didn't duplicate a node (same id) that exist deeper in the graph
            seen = set()
            dupes = []
            for path, n in self.all_nodes.items():
                if n in seen:
                    dupes.append([path, n])
                else:
                    seen.add(n)

            if dupes:
                match = [[path, n]
                         for path, n in before.items() if n in dupes[0]]
                message = f'Duplicate nodes found {dupes[0][0]} and {match[0][0]}'
                raise ImproperTensorGraphError(message)

        return node

    def add_node(self, node: Union[Callable, TensorGraph]):
        """
        Add a node to the HierarchalTensorGraph.

        Parameters
        ----------
        node : Callable, or HierarchalTensorGraph
            If a callable is passed, it is wrapped into an HTG and added to
            the graph if it doesn't yet exist. Otherwise, if an HTG is passed,
            it is added to the graph if it doesn't yet exist.

        Raises
        -------
        ImproperTensorGraphError
            If not callable or a HierarchalTensorGraph.

        """

        self.logger.info(f'Adding node {node} to HTG {self.name}')

        if not (callable(node) or isinstance(node, HierarchalTensorGraph)):
            message = f'A node must be either callable or a HierarchalTensorGraph'
            raise ImproperTensorGraphError(message)

        # we'll use get_node to add it to the graph. get_node will check if exist
        # and only add it if does not. It also checks node meets various criteria before
        # adding it to the graph.
        self.get_node(node)

    def add_io(self, node: Union[str, Callable]):
        """ Add input/output edges and (optionally) input/output specs to a node """

        self.logger.info(f'Adding i/o to node {node} in HTG {self.name}')

        self.add_edge('input', node)
        self.add_edge(node, 'output')
        if hasattr(node, 'input_spec'):
            self.get_node(node).inputs = node.input_spec
        if hasattr(node, 'output_spec'):
            self.get_node(node).outputs = node.output_spec

    def remove_node(self, node):
        """
        Removes a node from the HierarchalTensorGraph

        """

        self.logger.info(f'Removing node {node} from HTG {self.name}')

        # remove node from string
        if isinstance(node, str):
            if node in self.nodes:
                self.graph.remove_node(node)
                return
            else:
                message = f'Node with name {node} not found'
                raise ImproperTensorGraphError(message)

        # remove node from HierarchalTensorGraph
        if isinstance(node, HierarchalTensorGraph):
            if not node.name in self.nodes:
                message = f'Node with name {node} not found'
                raise ImproperTensorGraphError(message)

            if not self[node.name] is node:
                message = f'Node with the name {node.name} exist but does match the one passed'
                raise ImproperTensorGraphError(message)

            self.graph.remove_node(node.name)
            return

        # remove node from basenode function
        # check if basenode with this name exist and if it has the same node value.
        # if it has the same node value then this refers to that basenode.
        if callable(node):
            name = HierarchalTensorGraph.get_name(node)
            if not name in self.graph:
                message = f'Node with name {node} not found'
                raise ImproperTensorGraphError(message)

            if not node is self[name].node:
                message = f'Node with the name {node.name} exist but does match the one passed'
                raise ImproperTensorGraphError(message)

            self.graph.remove_node(name)
            return

        # if not one of the above, throw an exception
        message = f'Unrecognized node type. Must be of type str, callable, or HierarchalTensorGraph'
        raise ImproperTensorGraphError(message)

    def replace_node(self, node: Union[str, Callable], new_node: Union[str, Callable]):
        pass

    def make_recurrent(self, node: Union[str, Callable], recurrence: int = 1):
        """
        Replaces a node with a new node in the HierarchalTensorGraph

        """

        if isinstance(node, str):
            node = self.get_node(node)
        elif (not node in self):
            message = f'Node with name {node} not found'
            raise ImproperTensorGraphError(message)

        if (recurrence <= 0):
            message = f'Recurrence must be a greater than 0'
            raise ImproperTensorGraphError(message)

        node_names = [node.name + '_' + str(index)
                      for index in range(recurrence)]

        # create duplicate nodes
        nodes = []
        for recurr_node in node_names:
            htg_copy = copy.deepcopy(node)
            htg_copy.name = recurr_node
            nodes.append(htg_copy)

        node.node = node
        node.add_edge('input', nodes[0])
        node.add_edges_from([(nodes[i], nodes[i+1])
                            for i in range(recurrence - 1)])
        node.add_edge(nodes[-1], 'output')

        node.is_recurrent = True

        return node

    def add_edge(self, source: Union[str, Callable], target: Union[str, Callable], rollout_axis: Union[int, None] = None, **attr):
        """

        Add an edge between the source and target nodes.

        Parameters
        ----------

        source : str, Callable, HierarchalTensorGraph
             The source from which to begin the edge

        target : str, Callable, HierarchalTensorGraph
             The sink from which to end the edge

        Raises
        ------
        ImproperTensorGraphError
            - If an edges is added to a basenode
            - If 'input' is passed as a sink
            - If 'output' is passed as a source
            - If a cyclic graph is created by adding the edge

        """

        self.logger.info(f'Adding edge from {source} to {target} in HTG {self.name}')

        if isinstance(source, str):
            if source == 'output':
                message = f'Output cannot be used as a source'
                raise ImproperTensorGraphError(message)

        if isinstance(target, str) & (self.node is self):
            if target == 'input':
                message = f'Input cannot be used as a target'
                raise ImproperTensorGraphError(message)

        if self.node is not self:
            message = f'It is not permitted to add edges to a basenode,'
            message += 'i.e. an HTG initialized with a callable node'
            raise ImproperTensorGraphError(message)

        source = self.get_node(source)
        target = self.get_node(target)

        if (source is not None) and (target is not None):
            self.graph.add_edge(source.name, target.name, **attr)

            self.graph.set_edge_attributes(
                {(source.name, target.name): {'rollout_axis': rollout_axis}})

            if (source == target):
                source.is_recurrent = True

    def add_edges_from(self, ebunch: list):
        """

        Parameters
        ----------

        ebunch: a list of tuples (source,target, attr)

        """

        self.logger.info(f'Adding edges from {ebunch} in HTG {self.name}')

        for i in ebunch:
            self.add_edge(*i)

    def __repr__(self):
        """ Represent the HierarchalTensorGraph """

        self.logger.info(f'Representing HTG {self.name}')

        return f'HierarchalTensorGraph("{self.name}", id={id(self)})'

    def __getitem__(self, path: str | tuple | TensorGraph) -> 'HierarchalTensorGraph':
        """ Retrieve the node which has the given name from our graph

        Parameters
        ----------
        path : string, tuple, or HierarchalTensorGraph
             In the case of a string, the name of the child node within the parent HTG.
             In the case of a path, the node with the given path relative to the parent HTG.
             In the case of a HierarchalTensorGraph, it checks for a match to all nodes
             within the graph.

        Returns
        ----------

        A HierarchalTensorGraph

        Raises
        -------

        ImproperTensorGraphError:


        """

        self.logger.info(f'Getting node {path} in HTG {self.name}')

        if isinstance(path, HierarchalTensorGraph):
            htg = list(filter(lambda x: x == path, self.all_nodes.values()))
            if htg:
                if len(htg) > 1:
                    raise ImproperTensorGraphError(
                        f'multiple nodes found for {path} in __getitem__')
                return htg[0]
            raise ImproperTensorGraphError(
                f'HTG {path} is not contained within HTG {self.name}')

        if not path:
            return self
        if isinstance(path, str):
            # if path == self.name: return self
            # return i/o nodes
            if path in ['input'] and (path not in self.graph):
                return HierarchalTensorGraph.input_node(path)

            if path in ['output'] and (path not in self.graph):
                return HierarchalTensorGraph.output_node(path)
            try:
                return self.graph.nodes[path]['htg']
            except:
                raise ImproperTensorGraphError(
                    f'node with path = {path} not in graph.')

        try:
            return self[path[0]][path[1:]]
        except:
            raise ImproperTensorGraphError(
                f'node with path = {path} not in graph.')

    def __contains__(self, path: str | tuple | TensorGraph) -> bool:
        """ Check if a node with the given name is in our graph.

        Parameters
        ----------
        path : string, tuple, or HierarchalTensorGraph
              In the case of a string, the name of the child node within the parent HTG.
              In the case of a path (tuple), the node with the given path relative
              to the parent HTG.

        """

        self.logger.info(f'Checking if node {path} is in HTG {self.name}')

        # if given a string check if it is direct child of node
        if isinstance(path, str):
            return (path == self.name) or self.graph.has_node(path)

        if isinstance(path, tuple):
            return path in self.all_nodes

        if isinstance(path, HierarchalTensorGraph):
            return path in self.all_nodes.values()

        raise ImproperTensorGraphError(
            'must pass either a str, tuple, or HierarchalTensorGraph')

    def feature_map(self, X: dict, io: str) -> dict:
        """
        Applies a map on the incoming and outgoing dictionaries.
        according to the specified HTG paramerters inputs/outputs
        and inputs/outputs_map.

        Parameters
        ----------

        X : dictionary to be mapped.

        io: 'input' or 'output' specifiying which to apply

        Raises
        ------

        ImproperTensorGraphError:
         - if the key is not found.
         - if multiple keys (coming from different nodes)
         of that name are found. In this case,
         the found keys will be given in the error message

        """

        self.logger.info(f'Applying feature map to {io} in HTG {self.name}')

        # flatten nested dict of inputs
        def flatten_dict(d, key):
            for k, v in d.items():
                if isinstance(k, str):
                    k = (k,)
                if isinstance(v, dict):
                    yield from flatten_dict(v, key + k)
                else:
                    yield [key + k, v]

        # find keys and replace them
        def find_and_replace(namelist: dict, X: dict) -> dict:
            """Tries to find keys specified in namelist and replace them with the values in namelist"""
            for k, v in namelist.items():

                # ignore identical replacements
                if k == v:
                    continue

                # if the name to be replaced already exists you cannot make the replacement
                if v in X.keys():
                    message = f'Cannot rename {k} as {v} because the name {v} already exists in {X}'
                    raise ImproperTensorGraphError(message)

                # exact match
                if k in X.keys():
                    X[v] = X.pop(k)
                    continue

                # look for partial matches in long names
                partials = [i for i in X.keys() if k in (i[-1],)]

                if not partials:
                    message = f'Could not find key = {k} to rename'
                    raise ImproperTensorGraphError(message)

                if len(partials) > 1:
                    message = f'Found multiple keys = {partials} for {k}. '
                    message += f'You need to use one of these tuples to specifiy it uniquely and rename it.'
                    raise ImproperTensorGraphError(message)

                X[v] = X.pop(partials[0])
            return X

        def find_and_select(namelist: list, X: dict) -> dict:
            """tries to find and select the subset of X with keys = namelist """

            x = {}
            for name in namelist:
                # exact matches
                if name in X.keys():
                    x[name] = X[name]
                    continue

                if isinstance(name, str) and (name,) in X.keys():
                    x[name] = X[(name,)]
                    continue

                # if not exact, look for a match in the last name in tuple keys
                partials = [i for i in X.keys() if name in (i[-1], 1)]

                if not partials:
                    message = f'In HierarchalTensorGraph[{self.name}] could not find key = {name}. '
                    message += f'You will have to rename an incoming key '
                    message += f'from the available keys = {X.keys()}'
                    raise ImproperTensorGraphError(message)

                if len(partials) > 1:
                    message = f'In HierarchalTensorGraph[{self.name}] found '
                    message += f'multiple keys = {partials} for key = {name}. '
                    message += f'You will have to rename an incoming key '
                    message += f'from the available keys = {X.keys()}.'
                    raise ImproperTensorGraphError(message)

                x[name] = X[partials[0]]
            return x

        # Ignore input/output nodes in multinode HTGs
        if self.name in ['input', 'output']:
            return X

        # Renaming and selecting inputs
        if io == 'input':
            if not (self._inputs_map or self.inputs):
                return X
            # flatten/convert to tuples
            x = dict([i for i in flatten_dict(X, ())])
            # rename
            if self._inputs_map:
                x = find_and_replace(self._inputs_map, x)
            # select
            if self.inputs:
                x = find_and_select(self.inputs, x)

        # Renaming and selecting outputs
        if io == 'output':
            self.output = None  # Clear cached output
            if not (self._outputs_map or self.outputs):
                self.output = X.copy()  # Set node output
                return X

            # flatten/convert to tuples
            x = dict([i for i in flatten_dict(X, ())])
            # select
            if self.outputs:
                x = find_and_select(self.outputs, x)
                self.output = x.copy()

            # If no selection occured set output before renaming.
            if not self.outputs:
                self.output = X.copy()

            # rename
            if self._outputs_map:
                x = find_and_replace(self._outputs_map, x)

        return x

    def __call__(self, X: dict) -> dict:
        """Propagate the given input through the graph.

        Parameters
        ----------

        X : dict
            Input of the HTG.

        Returns
        -------

        Outputs produce by executing the underlying graph in the
        form {path: output value}.

        Raises
        ------

        ImproperTensorGraphError
            - If input and output nodes do not exists

        """

        self.logger.info(f'Calling HTG {self.name}')

        # if not a basenode do some error checking
        if not self.is_basenode:

            # check that i/o exist
            if not all([b in self.graph for b in ['input', 'output']]):
                message = f'A HierarchalTensorGraph name must contain i/o nodes'
                raise ImproperTensorGraphError(message)

            # check that only i/o is a source/sink
            sources = list(filter(lambda x: x != 'input', self.sources))
            if sources:
                message = f'Only input can be a source node in the graph. '
                message += f'Found sources {[self[i] for i in sources]}'
                raise ImproperTensorGraphError(message)

            sinks = list(filter(lambda x: x != 'output', self.sinks))
            if sinks:
                message = f'Only output can be a sink in the graph. '
                message += f'Found sinks {[self[i] for i in sinks]}'
                raise ImproperTensorGraphError(message)

        _X = self.feature_map(X, 'input')

        # Removes 'input' or 'output' nesting of keys
        def flatten(d):
            # find 'input' or 'output' items that are dicts
            keys = [k for k in ['input', 'output']
                    if k in d and isinstance(d[k], dict)]

            # If found, flatten
            if (keys):
                return [d.update(d.pop(k, {})) for k in ['input', 'output']] and d

            # otherwise, return original
            return d

        # Basenode call
        if self.is_basenode:
            return self.feature_map(self.node(flatten(_X)), 'output')

        # Recursive case: traverse graph in reverse, from output to input
        def nodes(name):
            return dict(
                self.graph.in_edges(name))  # All input nodes

        def search(name, depth):
            output_dict = {}
            rollout_dict = {}

            # @TODO: It might be worth it to create a cache for ALL nodes
            # such that that we don't have to specify how the data
            # gets cached based on whether the node is recurrent or not.
            # the cache logic would have to live in the respective function for caching.
            for input_node in nodes(name):
                if self.graph.edges[input_node, name]['rollout_axis'] is not None:
                    rollout_dict[input_node] = self[input_node].roll_out
                else:
                    tmp_depth = depth + (input_node == name)
                    input_name, output_value = traverse(
                        input_node, tmp_depth)
                    output_dict[input_name] = output_value

            if (len(rollout_dict)):

                outputs = []
                names, rollouts = zip(*rollout_dict.items())
                for values in zip(*rollouts):
                    rollout_values = dict(zip(names, values))
                    rollout_values = {k: v for k, v in rollout_values.items()}

                    outputs.append(output_dict | rollout_values)

                var = [flatten(self[name](o or _X)) for o in outputs]

                var_map = {k: [{x: y for x, y in o[k].items() if hasattr(
                    y, '__len__') or y is not None} if isinstance(o[k], dict) else o[k] for o in var if o[k] is not None] for k in var[0]}
                var_map = {k: [x for x in v if not isinstance(x, dict) or len(x) > 0] for k, v in var_map.items()}

                return var_map
            else:
                return self[name](output_dict or _X)

        traverse = cache(lambda name, depth=0: (name, (search(name, depth))))

        # can loop through and define
        for node in self.nodes:
            if not hasattr(self[node], 'roll_out'):
                for input_node in nodes(node):
                    if self.graph.edges[input_node, node]['rollout_axis'] is not None:
                        rollout_axis = self.graph.edges[input_node,
                                                        node]['rollout_axis']
                        self[input_node].roll_out = Recurrence(
                            self, input_node, traverse, _X, rollout_axis)

        return self.feature_map(flatten(dict(map(traverse, self.sinks))), 'output')

    def expand_graph_node(self, nodename: str, g=None):
        """ expands the graph of nodename and returns a new graph with the expansion

        Parameters:

        nodename : str
             Name of the node to expand.

        g: BaseGraph
             Graph from which to expand the node. If None, uses
             self.graph.

        Returns: A new graph with the expanded node.

        """

        self.logger.info(f'Expanding node {nodename} in HTG {self.name}')

        if not g:
            graph = self.graph.copy()
        else:
            graph = g.copy()

        # if basenode cannot be expanded
        if self[nodename].is_basenode:
            return graph

        # in/out edges of node
        in_node_edges = graph.in_edges(nodename)
        out_node_edges = graph.out_edges(nodename)

        # input/output edges of within the node
        edges_within_node = self[nodename].edges
        in_edges = [e for e in edges_within_node if 'input' in e]
        out_edges = [e for e in edges_within_node if 'output' in e]

        # create edges by fusing edges and adding parent to the names
        create_edges = [(x[0], nodename + '.' + y[1])
                        for x in in_node_edges for y in in_edges]
        create_edges += [(nodename + '.' + x[0], y[1])
                         for x in out_edges for y in out_node_edges]
        create_edges += [(nodename + '.' + x[0], nodename + '.' + x[1]) for x in
                         [e for e in edges_within_node if ('input' not in e) and (not 'output' in e)]]
        subnodes = {nodename + '.' + k: node for (k, node) in self[nodename].nodes.items()
                    if k not in ['input', 'output']}

        # copy graph and remove node
        graph.remove_node(nodename)

        # add in new edges and subnodes
        for k, v in subnodes.items():
            graph.add_node(k, htg=v)

        graph.add_edges_from(create_edges)

        return graph

    def _get_node(self, name, node=None):
        """ Get the node with the given name

        Parameters:

        name : str
                The name of the node to get

        node : HierarchalTensorGraph

        Returns: The node with the given name

        """

        self.logger.info(f'Getting node {name} in HTG {self.name}')
        
        node = self if node is None else node

        # if node is in current graph
        if (not '.' in name):
            return node[name]

        # if node is in a subgraph, recurse
        ind = name.split('.', 1)[0]
        name = name.split('.', 1)[1]
        return self._get_node(name, node[ind])

    def _get_parent(self, source, node=None):
        """ Get the parent of the current node

        Parameters:

        source : str
                The source node to get the output from

        Returns: The parent node

        """

        self.logger.info(f'Getting parent of node {source} in HTG {self.name}')

        node = self if node is None else node

        # if parent is in current graph
        if (source.count('.') == 1):
            ind = source.split('.', 1)[0]
            return node[ind]
        elif (not '.' in source):
            return self

        # if parents is in a subgraph, recurse
        ind = source.split('.', 1)[0]
        source = source.split('.', 1)[1]

        return self._get_parent(source, node[ind])

    def edge_label(self, source, target):
        """ Returns the edge label between source and target nodes

        Parameters:

        source : str
                The source node to get the output from

        target : str
                The target node to get the input from

        Returns: A dictionary with the edge label

        """

        self.logger.info(f'Getting edge label from {source} to {target} in HTG {self.name}')

        # get source and target nodes
        source_node = self._get_node(source)
        target_node = self._get_node(target)

        # get source outputs
        tmp_output = source_node.outputs if not source == 'input' else self.inputs
        tmp_output = self.get_outputs(self._get_parent(source)) if (
            '.' in source and not '.' in target) else tmp_output

        # convert source output format to list of labels
        if isinstance(tmp_output, dict):
            source_outputs = list(tmp_output.keys())
        else:
            source_outputs = tmp_output

        # get source output correspecting to the mapped output
        if len(source_node._outputs_map) > 0 or len(target_node._inputs_map) > 0:
            tmp_outputs = []
            for k in source_outputs:
                if k in source_node._outputs_map:
                    tmp_outputs.append(source_node._outputs_map[k])
                elif k in target_node._inputs_map:
                    tmp_outputs.append(target_node._inputs_map[k])
                else:
                    tmp_outputs.append(k)

            source_outputs = tmp_outputs

        # get target inputs
        tmp_input = target_node.inputs if not target == 'output' else self.outputs

        # convert target input format to list of labels
        if isinstance(tmp_input, dict):
            target_inputs = list(tmp_input.keys())
        else:
            target_inputs = tmp_input

        # get target input correspecting to the mapped input
        if len(source_node._outputs_map) > 0 and len(target_node._inputs_map) > 0:
            tmp_inputs = []
            for k in target_inputs:
                if k in target_node._inputs_map:
                    tmp_inputs.append(target_node._inputs_map[k])
                elif k in source_node._outputs_map:
                    tmp_inputs.append(source_node._outputs_map[k])
                else:
                    tmp_inputs.append(k)

            target_inputs = tmp_inputs

        # return dictionary of edge labels
        return {(source, target): (source_outputs, target_inputs)}

    def get_all_edge_labels(self, graph):
        """ Returns all edge labels in the graph

        Parameters:

        graph : BaseGraph
                The graph to get the edge labels from

        Returns: A dictionary with the edge labels

        """

        self.logger.info(f'Getting all edge labels in HTG {self.name}')

        edge_labels = {}
        avial = {}
        req = {}

        # for all edges in the graph
        for edge in graph.edges:
            source = edge[0]
            target = edge[1]

            # get edge labels
            edge_labels.update(self.edge_label(source, target))

            # get available inputs
            if target in avial:
                if source == 'input':
                    avial[target] += edge_labels[(source, target)][1]
                else:
                    avial[target] += edge_labels[(source, target)][0]
            else:
                if source == 'input':
                    avial[target] = edge_labels[(source, target)][1]
                else:
                    avial[target] = edge_labels[(source, target)][0]
                if source == 'input':
                    avial[target] += edge_labels[(source, target)][1]
                else:
                    avial[target] += edge_labels[(source, target)][0]

            # get required inputs
            if target in req and not source == 'input':
                req[target] += edge_labels[(source, target)][1]
            else:
                req[target] = edge_labels[(source, target)][1]

        validity = {}
        # check if all required inputs are available
        for tar in avial.keys():
            avial[tar] = list(set(avial[tar]))
            req[tar] = list(set(req[tar]))

            # determine validity color based on matching between required and available inputs
            if all(x in avial[tar] for x in req[tar]):
                validity[tar] = True
            else:
                validity[tar] = False

        # colors = mcolors.CSS4_COLORS
        colors = mcolors.XKCD_COLORS
        # colors = mcolors.TABLEAU_COLORS

        edge_attr = {}
        for edge in graph.edges:
            source = edge[0]
            target = edge[1]

            random = rand.randint(1, len(list(colors.keys())))

            edge_attr[(source, target)] = {
                'validity': validity[target],
                'label': edge_labels[(source, target)],
                'source-color': list(colors.keys())[random]}

        return edge_attr

    def print_edge_labels(self, expand_nodes=None):
        """ Prints all edge labels in the graph

        Parameters:

        expand_nodes : str, list, or 'all'

        """

        self.logger.info(f'Printing all edge labels in HTG {self.name}')

        # check if nodes are empty
        def is_not_all_empty(graph):
            for v in graph:
                if not graph.nodes[v]['htg'].is_basenode:
                    return True
            return False

        g = self.graph

        if expand_nodes:
            if not isinstance(expand_nodes, list) and expand_nodes != 'all':
                for node in [expand_nodes]:
                    g = self.expand_graph_node(node, g)

            # Expand graph nodes and create new graph
            if isinstance(expand_nodes, list):
                for node in expand_nodes:
                    g = self.expand_graph_node(node, g)

            if expand_nodes == 'all':
                while is_not_all_empty(g):
                    nodes = g.get_node_attributes('htg')
                    for k, node in nodes.items():
                        if not node.is_basenode:
                            g = self.expand_graph_node(node.name, g)

        # get all edge labels and attributes
        edge_labels = self.get_all_edge_labels(g)
        label_output = []
        for k, v in edge_labels.items():
            outputs = list(set(v['label'][0]))
            inputs = list(set(v['label'][1]))
            validity = v['validity']
            label_output.append(
                {'source': k[0], 'target': k[1], 'source output': outputs, 'target inputs': inputs, 'valid': validity})

        # convert to dataframe and apply color
        label_output = pd.DataFrame(label_output)

        label_output = label_output.style.apply(lambda x: [
            'background-color: green' if x['valid'] else 'background-color: red' for v in x], axis=1)

        # return dataframe
        return label_output

    def draw(self, expand_nodes=None, layout='kamada_kawai_layout'):
        """ Draws the graph

        Parameters:

        expand_nodes : str, list, or 'all'

        layout : str

        Returns: A new graph with the expanded node.

        """

        self.logger.info(f'Drawing HTG {self.name}')

        # check if nodes are empty
        def is_not_all_empty(graph):
            for v in graph:
                if not graph.nodes[v]['htg'].is_basenode:
                    return True
            return False

        g = self.graph

        if expand_nodes:
            if not isinstance(expand_nodes, list) and expand_nodes != 'all':
                for node in [expand_nodes]:
                    g = self.expand_graph_node(node, g)

            # Expand graph nodes and create new graph
            if isinstance(expand_nodes, list):
                for node in expand_nodes:
                    g = self.expand_graph_node(node, g)

            if expand_nodes == 'all':
                while is_not_all_empty(g):
                    nodes = g.get_node_attributes('htg')
                    for k, node in nodes.items():
                        if not node.is_basenode:
                            g = self.expand_graph_node(node.name, g)

        # Draw the final graph
        if layout == 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'spetral_layout':
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = None

        # get all edge labels and attributes
        edge_attributes = self.get_all_edge_labels(g)
        edge_styles = ['solid' if v['validity'] else 'dashed' for k, v in edge_attributes.items()]
        edge_colors = [v['source-color'] for k, v in edge_attributes.items()]

        # draw
        nx.draw(g, pos, edge_color=edge_colors, style=edge_styles, with_labels=True, alpha=1, font_size=10, node_size=1000,
                node_color='white', font_color='darkblue', font_family='Impact')


class Recurrence:
    """
    Define recurrence for a given node in an HTG
    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __init__(self, node, name, traverse, X, rollout_axis):
        self.cache = {}
        self.node = node
        self.name = name
        self.traverse = traverse
        self.X = X
        self.rollout_axis = rollout_axis

        self.logger.info(f'Initializing Recurrence for node {name} in HTG {node.name}')

    def __iter__(self):
        i = 0
        while True:
            try:
                yield self[i]
                i += 1
            except StopIteration:
                return

    # Dimension for any node that is being rolled out
    # is always assumed to be the same for any of the nodes edges.
    # This assumption will need to evaluated and changed in
    # the future.
    def __getitem__(self, index):

        self.logger.info(f'Getting item {index} in HTG {self.name}')
        if not self.node[self.name].is_recurrent:
            values = self.traverse(self.name)[
                1] if self.name != 'input' else self.X

            if any(isinstance(v, list) for v in values.values()):
                max_iter = min([len(v) for v in values.values()])
            else:
                max_iter = min([v.shape[self.rollout_axis]
                               for v in values.values()])

            if index < max_iter:
                if not any(isinstance(v, list) for v in values.values()):
                    index = (slice(None), ) * self.rollout_axis + (index, )
                return {k: v[index] for k, v in self.flatten(values).items()}

            raise StopIteration('No more values')

        if index == 0:
            self.cache[index] = {k: None for k in self.node[self.name].outputs}

        elif index not in self.cache:
            input_values = {}

            # index is not yet computed
            for input_node in dict(self.node.graph.in_edges(self.name)):
                
                # if the input node is recurrent
                if self.node.graph.edges[input_node, self.name]['rollout_axis'] is not None:
                    recurrence = self.node[input_node].roll_out
                    input_values[input_node] = recurrence[index - 1]
                else:
                    
                    # get the value of the input node
                    _, output_value = self.traverse(input_node)
                    input_values[input_node] = output_value

            self.cache[index] = self.node[self.name](input_values)

        return self.cache[index]

    # Removes 'input' or 'output' nesting of keys
    def flatten(self, d):
        # find 'input' or 'output' items that are dicts
        keys = [k for k in ['input', 'output']
                if k in d and isinstance(d[k], dict)]

        # If found, flatten
        if keys:
            return [d.update(d.pop(k, {})) for k in ['input', 'output']] and d

        # otherwise, return original
        return d
