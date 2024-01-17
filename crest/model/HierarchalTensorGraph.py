from collections.abc import Callable
from functools import cache
from typing import Union
from copy import deepcopy

# networkx breaks numpy if it's not also loaded, due to nx.lazy_import modifying sys modules
import numpy as np 
import networkx as nx
import marshal, base64
import traceback

from .TensorGraph import TensorGraph, ImproperTensorGraphError
from .graphs.NetworkXGraph import NetworkXGraph


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

    def __init__(self, 
        node    : None | Callable = None, 
        name    : None | str = None, 
        inputs  : dict = {}, 
        outputs : dict = {},
    ):

        self.name = name or HierarchalTensorGraph.get_name(node)
        self.node = node or self
        self.graph = NetworkXGraph()
        self.output = []
        self.inputs = inputs
        self.outputs = outputs
        self._inputs_map = {}  # dictionary specifying input feature renaming
        self._outputs_map = {}  # dictionary specifying input feature renaming

        # check that the HTG has a name
        if self.name == 'None':
            message = f'A HierarchalTensorGraph name must be given, but a {self.name} was found'
            raise ImproperTensorGraphError(message)

        # check that if node is passed it is callable
        if (self.node is not self) and (not callable(self.node)):
            message = f'node must be callable'
            raise ImproperTensorGraphError(message)

    def __iter__(self):
        """ Iterate through all nodes within the HTG """

        def traverse_nodes(graph, path):
            """ Recursive generator """
            for node in graph:
                yield [path + (graph.nodes[node]['htg'].name,), graph.nodes[node]['htg']]
                if not graph.nodes[node]['htg'].is_empty:
                    yield from traverse_nodes(graph.nodes[node]['htg'].graph, path + (graph.nodes[node]['htg'].name,))

        yield from traverse_nodes(self.graph, ())

    # for pickling

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    @staticmethod
    def serialize_callable(node: Callable):
        """

        Serialize the node object based on the type we have inferred.
        If the type extends HierarchalTensorGraph, we call its serialize function. 

        Parameters
        ----------
        obj : Callable, extends HierarchalTensorGraph | Marshal
             The object from which we determine node type.

        """

        try:
            # print(type(node), node.__class__)
            if (hasattr(node, 'serialize') and not node.__class__.__name__ == 'HierarchicalTensorGraph'):
                return {'module': str(node.__class__.__module__), 'name': str(node.__class__.__name__), 'obj': node.serialize()}
            else:
                def LAMBDA(): return 0
                if isinstance(node, type(LAMBDA)) and 'lambda' in node.__name__:
                    lambda_serialized = marshal.dumps(node.__code__)
                    # print('lambda serialized')
                    return {'module': 'lambda', 'name': 'lambda', 'obj': base64.b64encode(lambda_serialized).decode('utf-8')}
                else:
                    raise ImproperTensorGraphError(
                        f'Unsupported base-node type')
        except:
            raise ImproperTensorGraphError(
                f'Could not find serialization function for base-node type')

    @staticmethod
    def deserialize_callable(data: dict):
        """

        Deserialize the node object based on the type we have inferred.
        If the type extends HierarchalTensorGraph, we call the deserialize function. 

        Parameters
        ----------
        dict : data, extends HierarchalTensorGraph | Marhsal
             The object from which we determine node type and deserialize.

        """

        try:
            modulename = data['module']
            classname = data['name']

            # print(classname, modulename)

            if ('lambda' in modulename):
                decoded_str = base64.b64decode(data['obj'])
                marshal_obj = marshal.loads(decoded_str)
                x = lambda x: eval(marshal_obj)
                return x
            else:
                module = __import__(modulename)
                class_ = getattr(module, classname)
                instance = class_()

                if hasattr(instance, 'deserialize'):
                    x = instance.deserialize(data['obj'])
                    return x
                else:
                    raise ImproperTensorGraphError(
                        f'Unsupported base-node type')

        except:
            raise ImproperTensorGraphError(
                f'Need deserialize function for base-node type')

    def equals(self, node: Callable) -> bool:
        """ Check if the given node is the same as this node """

        if (self.node and self.node.is_empty):
            if (node and node.is_empty):

                # compare names because it is impossible to compared callables
                return (self.node.name == node.name)
            else:
                return False

        if (self.node.graph.is_isomorphic(node.graph.graph)):
            return True

        return False

    def to_json(self):
        """

        Enables the serialization of the network graph associated with this node. 

        """

        try:
            # create a copy of the current node and attributes

            d = deepcopy(self.__dict__)
            d['graph'] = None
            d['node'] = HierarchalTensorGraph.serialize_callable(self.node) if (
                self.node and not isinstance(self.node, HierarchalTensorGraph)) else None

            # iterate through nodes in the graph and recursively serialize them
            d['nodes'] = {}
            for node in list(self.graph.graph.nodes):
                n = self.get_node(node)
                d['nodes'][node] = HierarchalTensorGraph.serialize_callable(
                    n.node) if (n.is_empty) else n.to_json()

            # store edges of connection to this node
            d['edges'] = list(self.graph.graph.edges)

            # return dictionary
            return d
        except Exception as e:
            # log for later debugging
            print(traceback.format_exc())
            return None

    def from_json(self, data):
        """

        Enables the deserialization of the network graph represented in a JSON structure. 

        Parameters:
        -----------
        data : dict
             The JSON object to deserialize.

        """
        try:
            # get name of the current node from input dictionary
            self.name = data['name']

            # get node and graph from input dictionary
            self.node = self if ('node' in data.keys(
            ) and data['node'] == None) else HierarchalTensorGraph.deserialize_callable(data['node'])
            self.graph = NetworkXGraph()

            # iterate through nodes in the graph and recursively deserialize them
            for k, v in data['nodes'].items():
                if isinstance(v, dict) and ('node' in v.keys()):
                    htg = HierarchalTensorGraph(name=k)
                    self.graph.add_node(k, htg=htg.from_json(v))
                else:
                    c = HierarchalTensorGraph.deserialize_callable(v)
                    htg = HierarchalTensorGraph(
                        name=k, node=c)
                    self.graph.add_node(k, htg=htg)

            # add edges to the graph
            self.graph.add_edges_from(data['edges'])

            # pick up other attributes of the node
            self._inputs_map = data['_inputs_map']
            self._outputs_map = data['_outputs_map']
            self.inputs = data['inputs']
            self.outputs = data['outputs']
            self.output = data['output']

            return self
        except Exception as e:
            # log for later debugging
            print(traceback.format_exc())
            return None

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
        inputs_map  : None | dict = None, 
        outputs_map : None | dict = None, 
        node        : None | str | tuple  | TensorGraph = None,
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
        # tuples must be exact matches.
        if isinstance(name, tuple):
            return dict(filter(lambda t: name == t[0], self.all_nodes.items()))

        if (partial):
            return dict(filter(lambda s: name in ''.join(s[0]), self.all_nodes.items()))
        return dict(filter(lambda s: name in s[0], self.all_nodes.items()))

    @property
    def all_nodes(self) -> dict:
        """ Returns a dict of all child nodes within the parent """
        return dict([i for i in self])

    @property
    def edges(self):
        return self.graph.edges

    @property
    def nodes(self) -> dict:
        """ Returns a dict of nodes within the parent HTG """
        return self.graph.get_node_attributes('htg')

    @property
    def sources(self) -> list:
        """ Nodes with no incoming edges """
        def is_source(node): return self.graph.in_degree(node) == 0
        return list(filter(is_source, self.graph.nodes))

    @property
    def sinks(self) -> list:
        """ Nodes with no outgoing edges """
        def is_sink(node): return self.graph.out_degree(node) == 0
        return list(filter(is_sink, self.graph.nodes))

    @property
    def is_empty(self) -> bool:
        """ Check if graph is empty """
        return self.graph.is_empty

    @staticmethod
    def identity(name: str) -> 'HierarchalTensorGraph':
        """
        Create an the identity HTG.

        name : string
             The name of the HTG

        """
        return HierarchalTensorGraph(lambda x: x, name)

    def get_node(self, node: Union[str, Callable, TensorGraph]) -> 'HierarchalTensorGraph':
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

        if not (isinstance(node, str) or isinstance(node, HierarchalTensorGraph) or callable(node)):
            message = f'node must be either callable, a string, or a HierarchalTensorGraph'
            raise ImproperTensorGraphError(message)

        # fetch node by its name
        if isinstance(node, str):
            # add io node if first time called
            if (node in ['input', 'output']) and (node not in self):
                io = HierarchalTensorGraph.identity(node)
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

        if not (callable(node) or isinstance(node, HierarchalTensorGraph)):
            message = f'A node must be either callable or a HierarchalTensorGraph'
            raise ImproperTensorGraphError(message)

        # we'll use get_node to add it to the graph. get_node will check if exist
        # and only add it if does not. It also checks node meets various criteria before
        # adding it to the graph.
        self.get_node(node)

    def add_io(self, node: Union[str, Callable]):
        """ Add input/output edges and (optionally) input/output specs to a node """
        self.add_edge('input', node)
        self.add_edge(node, 'output')
        if hasattr(node, 'input_spec'):  self.get_node(node).inputs  = node.input_spec
        if hasattr(node, 'output_spec'): self.get_node(node).outputs = node.output_spec

    def remove_node(self, node):
        """
        Removes a node from the HierarchalTensorGraph

        """

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

    def add_edge(self, source: Union[str, Callable], target: Union[str, Callable], **attr):
        """

        Add an edge between the source and target nodess.

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

        if (isinstance(source, str)):
            if source == 'output':
                message = f'Output cannot be used as a source'
                raise ImproperTensorGraphError(message)

        if (isinstance(target, str) & (self.node is self)):
            if target == 'input':
                message = f'Input cannot be used as a target'
                raise ImproperTensorGraphError(message)

        if (self.node is not self):
            message = f'It is not permitted to add edges to a basenode,'
            message += 'i.e. an HTG initialized with a callable node'
            raise ImproperTensorGraphError(message)

        source = self.get_node(source)
        target = self.get_node(target)

        if (source is not None) and (target is not None):
            self.graph.add_edge(source.name, target.name)
            if not self.graph.is_directed_acyclic_graph:
                message = f'Adding this edge created a cyclic graph'
                raise ImproperTensorGraphError(message)

    def add_edges_from(self, ebunch: list):
        """
        Add multiple edges

        Parameters
        ----------

        ebunch: a list of tuples (source,target)

        """

        for i in ebunch:
            self.add_edge(*i)

    def __repr__(self):
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

        if (isinstance(path, HierarchalTensorGraph)):
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
        if (isinstance(path, str)):
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

        # apply feature map
        _X = self.feature_map(X, 'input')

        # if not a basenode
        if (self.node is self):

            # check that i/o exist
            if not all([b in self.graph for b in ['input', 'output']]):
                message = f'A HierarchalTensorGraph name must contain i/o nodes'
                raise ImproperTensorGraphError(message)

            # check that only i/o is a source/sink
            sources = list(filter(lambda x: x != 'input', self.sources))
            if sources:
                message = f'Only input can be a source node in the graph. '
                message += f'Found sources {[ self[i] for i in sources]}'
                raise ImproperTensorGraphError(message)

            sinks = list(filter(lambda x: x != 'output', self.sinks))
            if sinks:
                message = f'Only output can be a sink in the graph. '
                message += f'Found sinks {[ self[i] for i in sinks]}'
                raise ImproperTensorGraphError(message)

        # Flatten input/output keys in the dict
        def flatten(d): return [d.update(d.pop(k, {}))
                                for k in ['input', 'output']] and d

        # Basenode: graph is empty
        if self.is_empty:
            return self.feature_map(self.node(flatten(_X)), 'output')

        # Recursive case: traverse graph in reverse, from output to input
        def nodes(name): return dict(
            self.graph.in_edges(name))  # All input nodes

        def search(name): return dict(
            map(traverse, nodes(name)))  # Traverse all inputs

        def output(name): return self[name](
            search(name) or _X)  # Get output for a node

        traverse = cache(lambda name: (name, self[name](search(name) or _X)))
        return self.feature_map(flatten(dict(map(traverse, self.sinks))), 'output')

    def expand_graph_node(self, nodename: str, g=None):
        """ expands the graph of nodename and returns a new graph with the expansion

        Parameters:

        nodename : str
             Name of the node to expand.

        g: BaseGraph
             Graph from which to expand the node. If None, uses
             self.graph.

        """

        if not g:
            graph = self.graph.copy()
        else:
            graph = g.copy()

        # if basenode cannot be expanded
        if self[nodename].is_empty:
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

    def draw(self, expand_nodes=None, layout='kamada_kawai_layout'):

        # check if nodes are empty
        def is_not_all_empty(graph):
            for v in graph:
                if not graph.nodes[v]['htg'].is_empty:
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
                while (is_not_all_empty(g)):
                    nodes = g.get_node_attributes('htg')
                    for k, node in nodes.items():
                        if not node.is_empty:
                            g = self.expand_graph_node(node.name, g)

        # Draw the final graph
        if layout == 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'spetral_layout':
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = None

        nx.draw(g, pos, with_labels=True, alpha=1, font_size=10, node_size=1000,
                node_color='white', font_color='darkblue', font_family='Impact')
