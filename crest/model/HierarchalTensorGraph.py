import logging
import warnings
from functools import cached_property,wraps
from functools import cache
from typing import Union
from collections.abc import Callable
import inspect
import numpy as np
from prettytable.colortable import ColorTable, Themes
import tensorflow as tf
import networkx as nx
import dill
from .TensorGraph import TensorGraph, ImproperTensorGraphError
from .graphs.NetworkXGraph import NetworkXGraph
from .Registry import Registry

logger = logging.getLogger(__name__)

class HierarchalTensorGraph(TensorGraph):
    """
    HierarchalTensorGraph (HTG): The HTG is used to build hierarchal graphs
    of Nodes (or other HTGs).

    Parameters
    ----------

    name : str
       The name of the HTG which must be different than other nodes
       in the graph. If name is None, it will default to
       ['name', '__name__', '__qualname__'].

    **attr: dict
        The attributes of the HTG. Defaults:
        'recurrent' : False,
        'return_seq' : False,
        'initialization' : None,
        'roll_out' : None,

    """

    # Register of all HTG subclasses
    registry = Registry()

    # Optional name for registry
    registry_name = 'HierarchalTensorGraph'

    # Subclass bool
    is_subclass = False

    @property
    def logger(self) -> logging.Logger:
        """ logger """
        return logging.getLogger(__name__)

    def __new__(cls,*args,**kwargs):
        obj = super().__new__(cls)

        # Wrap build function
        og_build = obj.build

        @wraps(og_build)
        def build_wrapper(*args,**kwargs):

            # If not subclass do nothing
            if not obj.is_subclass:
                return

            # If already built, return
            if obj.attributes['built']:
                return

            og_build(*args,**kwargs)
            obj.attributes['built'] = True

        # Set new build function 
        obj.build = build_wrapper

        return obj

    def __init__(self,name: str,**attr):
        super().__init__()
        self.name = name
        self.node = self
        self.graph = NetworkXGraph()
        self.output = []
        self.inputs = {}
        self.outputs ={}

        # Default attributes for nodes
        self.attributes = {
            'recurrent' : False,
            'return_seq' : False,
            'roll_out' : None,
            'built' : None
        }

        if self.is_subclass:
            self.attributes['built'] = False

        # Overwrite default attributes
        for k,v in attr.items():
            self.attributes[k] = v

        # log the creation of the HTG
        self.logger.info("Created HierarchalTensorGraph %s",self.name)

    # Automatically register all subclasses
    # Runs once per subclass creation
    def __init_subclass__(cls,*args,**kwargs):

        # Set subclass to true
        cls.is_subclass = True

        # Set default registry name to class name
        basename = inspect.getmro(cls)[1].__name__
        if cls.registry_name == basename:
            cls.registry_name = cls.__name__

        cls.registry.register(cls)

    def initial_state(self,X):
        message = f'HierarchalTensorGraph  = {self.name} must implement '
        message += 'a initial_state function since it is recurrent'
        raise NotImplementedError(message)

    def build(self):
        """ Build functions for subclasses of HTG """

    def __iter__(self):
        """ Iterate through all nodes within the HTG """

        self.logger.info("Iterating through all nodes in HTG %s", self.name)

        def traverse_nodes(graph, path):
            """ Recursive generator """
            for node in graph:
                yield [path + (graph.nodes[node]['htg'].name,), graph.nodes[node]['htg']]
                if not graph.nodes[node]['htg'].is_basenode:
                    yield from traverse_nodes(graph.nodes[node]['htg'].graph, 
                                              path + (graph.nodes[node]['htg'].name,))

        yield from traverse_nodes(self.graph, ())

    def equals(self, node: Callable) -> bool:
        """ Check if the given node is the same as this node """

        self.build()

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

    def encode(self,type='dill',**kwargs):
        """ Serialization """
        self.build()

        encode = {
            'name' : dill.dumps(self.name,**kwargs),
            'edges' : dill.dumps(list(self.edges(data=True)),**kwargs)
        }

        attributes = {}
        for k,v in self.attributes.items():
            attributes[k] = dill.dumps(v,**kwargs)
        encode['attr'] = attributes

        # Collect the json dicts of nodes
        encode['nodes'] = []
        for name, node in self.nodes.items():
            if not name in ['input', 'output']:
                encode['nodes'].append({
                    'crest_registry_name' : node.registry_name,
                    **node.encode(type,**kwargs)
                    })

        return encode

    @classmethod
    def decode(cls,encode,type='dill',**kwargs):
        """ recursive deserializer """

        # Get nodes
        nodes = []
        for node in encode.pop('nodes'):
            obj = cls.registry[node.pop('crest_registry_name')]
            nodes.append(obj.decode(node,type,**kwargs))

        edges = dill.loads(encode.pop('edges'),**kwargs)

        attributes = {}
        for k,v in encode.pop('attr').items():
            attributes[k] = dill.loads(v,**kwargs)

        decode = {k : dill.loads(v,**kwargs) for k,v in encode.items()} 

        htg = cls(**decode)

        # Set attributes
        htg.attributes = attributes

        if not htg.graph.is_empty:
            return htg

        # Build graph 
        for i in nodes: 
            htg.add_node(i)

        for e in edges:
            s, t, a = e
            htg.add_edge(s, t, **a)

        return htg

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

        self.logger.info("Searching for node %s in HTG", self.name)

        # tuples must be exact matches.
        if isinstance(name, tuple):
            return dict(filter(lambda t: name == t[0], self.all_nodes.items()))

        if partial:
            return dict(filter(lambda s: name in ''.join(s[0]), self.all_nodes.items()))
        return dict(filter(lambda s: name in s[0], self.all_nodes.items()))

    @property
    def all_nodes(self) -> dict:
        """ Returns a dict of all child nodes within the parent """

        self.logger.info("Getting all nodes in HTG %s",self.name)

        return dict([i for i in self])

    @property
    def edges(self):
        """ returns edges in the graph """
        self.logger.info("Getting all edges in HTG %s",self.name)
        return self.graph.edges

    @property
    def nodes(self) -> dict:
        """ Returns a dict of nodes within the parent HTG """

        self.build()

        self.logger.info("Getting all nodes in HTG %s",self.name)

        return self.graph.get_node_attributes('htg')

    @property
    def sources(self) -> list:
        """ Nodes with no incoming edges """

        self.logger.info("Getting all sources in HTG %s",self.name)

        def is_source(node): 
            return self.graph.in_degree(node) == 0

        return list(filter(is_source, self.graph.nodes))

    @property
    def sinks(self) -> list:
        """ Nodes with no outgoing edges """

        self.logger.info("Getting all sinks in HTG %s",self.name)

        def is_sink(node): 
            return self.graph.out_degree(node) == 0

        return list(filter(is_sink, self.graph.nodes))

    @property
    def is_basenode(self) -> bool:
        """ Check if graph is a basenode """
        self.logger.info("Checking if HTG %s is a basenode",self.name)
        return False

    def get_node(self, node: Union[str, TensorGraph]):
        """ 
        Adds a node to the graph if it doesn't exist and
        does a variety of error checking before adding.
        Also adds io nodes on the first use.
        
        ""

        Parameters
        ----------
        node : str or HTG
            If a string is passed, the HTG (which must already exist in the graph) 
            is returned. If a HTG is passed,
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
            - If the same referenced HTG passed already exists in the HTG.
            - a string or a HierarchalTensorGraph
            
        """

        self.logger.info(f'Getting node {node} in HTG {self.name}')

        if not (isinstance(node, str) or isinstance(node, HierarchalTensorGraph)):
            message = 'node must be either a string, or a HierarchalTensorGraph'
            raise ImproperTensorGraphError(message)

        # fetch node by its name
        if isinstance(node, str):
            # add io node if first time called
            if (node in ['input', 'output']) and (node not in self):
                io = Identity(node)
                self.graph.add_node(io.name,htg=io,**io.attributes)
                return io

            # otherwise if using a str() it must already be in the graph
            assert (node in self), f'Unknown name "{node}"'
            return self[node]

        # check if you passed a different HTG with the same name
        nodes = self.graph.get_node_attributes('htg')
        if (node.name in self.graph) and (node is not nodes[node.name]):
            message = f'node ({node.name} = {node}) has the same name'
            message += f'(and path) as existing node {nodes[node.name]}'
            raise ImproperTensorGraphError(message)

        # if node is not in the graph add it
        if not node.name in self.graph:

            # Nodes before adding
            before = self.all_nodes
            
            # Add node to graph
            self.graph.add_node(node.name, 
                                htg=node,
                                **node.attributes
                               )

            # Check we didn't duplicate a node (same id) that exist deeper in the graph
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

    def add_node(self, node: TensorGraph):
        """
        Add a node to the HierarchalTensorGraph.

        Parameters
        ----------
        node :  HierarchalTensorGraph or Node(HTG)

        Raises
        -------
        ImproperTensorGraphError
             HierarchalTensorGraph.

        """

        self.logger.info(f'Adding node {node} to HTG {self.name}')

        if not isinstance(node, HierarchalTensorGraph):
            message = 'A node must be a HierarchalTensorGraph'
            raise ImproperTensorGraphError(message)

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

        # Get name
        name = node if isinstance(node,str) else node.name

        # Check if it's in the graph
        if name not in self.nodes:
            message = f'Node with name {node} not found'
            raise ImproperTensorGraphError(message)

        # Check it's the right node not some other with the same name
        if isinstance(node, HierarchalTensorGraph):
            if not self[node.name] is node:
                message = f'Node with the name {node.name} exist but does match the one passed'
                raise ImproperTensorGraphError(message)

        self.graph.remove_node(name)
        return

    def add_edge(
                 self, source: Union[str, Callable], 
                 target: Union[str, Callable], 
                 rollout_axis: Union[int, None] = None,
                 rename: dict[str,str] = {},
                 features: list[str] | str = [],
                 **attr
                 ):
        """

        Add an edge between the source and target nodes.

        Parameters
        ----------

        source : str, Callable, HierarchalTensorGraph
             The source from which to begin the edge

        target : str, Callable, HierarchalTensorGraph
             The sink from which to end the edge

        rename : dict[str,str]
            Dictionary to rename incoming keys with items = name : new_name

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
            if source == "output":
                message = 'Output cannot be used as a source'
                raise ImproperTensorGraphError(message)

        if isinstance(target, str) & (self.node is self):
            if target == "input":
                message = 'Input cannot be used as a target'
                raise ImproperTensorGraphError(message)

        if self.node is not self:
            message = 'It is not permitted to add edges to a basenode,'
            message += 'i.e. an HTG initialized with a callable node'
            raise ImproperTensorGraphError(message)

        # Default edge attributes
        default_attributes = {
            'rollout_axis': rollout_axis,
            'rename' : rename,
            'features' : features if isinstance(features,list) else [features],
        }

        source = self.get_node(source)
        target = self.get_node(target)

        # Why are we checking it's not None? 
        if (source is not None) and (target is not None):

            # Add or overwrite to default arguments 
            for k,v in attr.items(): default_attributes[k] = v

            # If target is a recurrent node and not set
            # add rollout_axis = 1
            if self.graph.get_node_attributes('recurrent')[target.name]:
                if default_attributes['rollout_axis'] is None:
                    default_attributes['rollout_axis'] = 1

            # Add edge
            self.graph.add_edge(source.name, target.name, **default_attributes)

            # Get all incoming inputs
            incoming = {}
            for e in list(self.graph.in_edges(target.name)):
                s = self.get_node(e[0])
                t = self.get_node(e[1])
                X = s.outputs

                if s.name == 'input':
                    X = t.inputs.copy()
                    if t.attributes['recurrent']:
                        for i in t.outputs:
                            if i in t.inputs:
                                X.pop(i)

                Y = self._edge_mapping(X,s.name,target.name)
                incoming[s.name] = Y

            # Keep track of incoming keys for non I/O nodes
            if(source.name != 'input' and target.name != 'output'):
                X = self._edge_mapping(source.outputs,source.name,target.name)
                for key in X:
                    for k,v in incoming.items():
                        if(key in v and k != source.name):
                            message = f'Cannot add_edge{(source.name,target.name)} becasue the'
                            message += f' key "{key}" is already being passed into'
                            message += f' node "{target.name}" through edge "{k}". You need to'
                            message += ' rename a feature or select which features'
                            message += ' to be passed in along these edges.'
                            raise ImproperTensorGraphError(message)

                # Check there's matching input/output keys
                subset = {k : v for k,v in X.items() if k in self[target.name].inputs}
                if not subset:
                    message = f'Cannot add_edge{(source.name,target.name)} becasue'
                    message += ' their are no matching output keys of'
                    message += f' node {source.name} in {target.name}.'
                    raise ImproperTensorGraphError(message)

                # Check tensor specs are consistent
                Y = self[target.name].inputs
                for k,v in subset.items():
                    consistent = True

                    # Check length of dims
                    consistent = len(v.shape) == len(Y[k].shape)
                    if not consistent:
                        message = 'Inconsistent tensor specs.'
                        message += f' From {source.name} found {k} = {v} '
                        message += f' and from {target.name} found {k} = {Y[k]}.'
                        raise ImproperTensorGraphError(message)

                    # Check dtype
                    consistent = v.dtype == Y[k].dtype
                    if not consistent:
                        message = 'Inconsistent tensor specs.'
                        message += f' From {source.name} found {k} = {v} '
                        message += f' and from {target.name} found {k} = {Y[k]}.'
                        raise ImproperTensorGraphError(message)

                    # Check dims
                    for d in zip(v.shape,Y[k].shape):
                        if d[1] is None:
                            continue
                        if d[0] == d[1]:
                            continue

                        message = 'Inconsistent tensor specs.'
                        message += f' From {source.name} found {k} = {v} '
                        message += f' and from {target.name} found {k} = {Y[k]}.'
                        raise ImproperTensorGraphError(message)


            # Build input spec
            if source.name == 'input':
                X = self._edge_mapping(target.inputs,source.name,target.name)

                # Remove recurrent inputs
                if self.graph.get_node_attributes('recurrent')[target.name]:
                    for k in target.outputs:
                        if k in X:
                            X.pop(k)

                for k,v in X.items():
                    self.inputs[k] = v

            # Build output spec
            if target.name == 'output': 
                # Check for duplicate keys in outpupt and use tuples if found
                X = self._edge_mapping(source.outputs,source.name,target.name)

                # Add keys to outputs
                for k,v in X.items():     
                    self.outputs[k] = v

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


    # find keys and replace them
    def find_and_replace(self,namelist: dict, X: dict) -> dict:
        """Tries to find keys specified in namelist and replace them with the values in namelist"""
        _X = X.copy()
        for k, v in namelist.items():

            # ignore identical replacements
            if k == v:
                continue

            # if the name to be replaced already exists you cannot make the replacement
            if v in _X.keys():
                message = f'Cannot rename {k} as {v} because the name {v} already exists in {X}'
                raise ImproperTensorGraphError(message)

            # exact match
            if k in _X.keys():
                _X[v] = _X.pop(k)
                continue

            # look for partial matches in long names
            partials = [i for i in _X.keys() if k in (i[-1],)]

            if not partials:
                message = f'Could not find key = {k} to rename'
                raise ImproperTensorGraphError(message)

            _X[v] = _X.pop(partials[0])

        return _X

    def find_and_select(self, X: dict, io: str | None = None) -> dict:
        """
        Flattens input/output and finds and selects the inputs/outputs
        specified in self.inputs/outputs.

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

        def _find_and_select(namelist: list, X: dict) -> dict:
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
                nm = name
                if isinstance(name, str): nm = (name,)
                partials = [i for i in X.keys() if nm == i[-len(nm):]]

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
            if not self.inputs:
                return X

            # flatten/convert to tuples
            x = dict([i for i in flatten_dict(X, ())])

            if self.inputs:
                x = _find_and_select(self.inputs, x)

        # Renaming and selecting outputs
        if io == 'output':
            self.output = None  # Clear cached output
            if not self.outputs:
                self.output = X.copy()  # Set node output
                return X

            # flatten/convert to tuples
            x = dict([i for i in flatten_dict(X, ())])

            # select
            if self.outputs:
                x = _find_and_select(self.outputs, x)
                self.output = x.copy()

        return x

    def _edge_mapping(self,
                      X : dict, 
                      source : str,
                      target : str
                     ):
        """
          Applies a mapping to the output of node across
          a particular outgoing edge.

          X : dict
            Output dict coming from the node

          source : str
              Name of the source node
              
          target : str
              Name of the source node

          build : bool
            Whether executing during graph building vs. execution
              
        """

        edge = self.graph.edges[source,target]
        _X = X.copy()

        # Select from incoming features
        if edge['features']:
            for i in edge['features']:
                if not i in _X:
                    message = f'Cannot select feature = {i} on edge {(source,target)}. '
                    message += f'Available keys are: {list(_X.keys())}'
                    raise ImproperTensorGraphError(message)

            _X = { k : v for k,v in _X.items() if k in edge['features']}

        # Rename features after selection if needed
        if edge['rename']:
            _X = self.find_and_replace(edge['rename'], _X)

        return _X


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
        # Check lazy build for subclassing
        self.build()

        self.logger.info(f'Calling HTG {self.name}')

        # if not a basenode do some error checking
        if not self.is_basenode:

            # check that i/o exist
            if not all([b in self.graph for b in ['input', 'output']]):
                message = f'HierarchalTensorGraph {self.name} must contain i/o nodes'
                raise ImproperTensorGraphError(message)

            # check that only i/o is a source/sink
            sources = list(filter(lambda x: x != 'input', self.sources))
            if sources:
                message = 'Only input can be a source node in the graph. '
                message += f'Found sources {[self[i] for i in sources]}'
                raise ImproperTensorGraphError(message)

            sinks = list(filter(lambda x: x != 'output', self.sinks))
            if sinks:
                message = 'Only output can be a sink in the graph. '
                message += f'Found sinks {[self[i] for i in sinks]}'
                raise ImproperTensorGraphError(message)

        _X = self.find_and_select(X, 'input')

        # Removes 'input' or 'output' nesting of keys
        def flatten(d):
            # find 'input' or 'output' items that are dicts
            keys = [k for k in ['input', 'output']
                    if k in d and isinstance(d[k], dict)]

            # If found, flatten
            if keys:
                return [d.update(d.pop(k, {})) for k in ['input', 'output']] and d

            # otherwise, return original
            return d

        # Basenode call
        if self.is_basenode:
            return self.find_and_select(self.node(flatten(_X)), 'output')

        # Recursive case: traverse graph in reverse, from output to input
        def nodes(name):
            return dict( self.graph.in_edges(name))  # All input nodes

        def search(name, depth):
            output_dict = {}
            rollout_dict = {}

            # @TODO: It might be worth it to create a cache for ALL nodes
            # such that that we don't have to specify how the data
            # gets cached based on whether the node is recurrent or not.
            # the cache logic would have to live in the respective function for caching.
            for input_node in nodes(name):
                if self.graph.edges[input_node, name]['rollout_axis'] is not None:
                    rollout_dict[input_node] = self[input_node].attributes['roll_out']
                else:
                    tmp_depth = depth + (input_node == name)
                    input_name, output_value = traverse(
                        input_node, tmp_depth)
                    output_dict[input_name] = self._edge_mapping(output_value,input_node,name)

            if len(rollout_dict):

                outputs = []
                names, rollouts = zip(*rollout_dict.items())
                for values in zip(*rollouts):
                    rollout_values = dict(zip(names, values))
                    # Apply edge mapping 
                    rollout_values = {k : self._edge_mapping(v,k,name) 
                                      for k,v in rollout_values.items()}

                    rollout_values = {k: v for k, v in rollout_values.items()}

                    outputs.append(output_dict | rollout_values)

                # Collect recurrent final output
                var = [flatten(self[name](o or _X)) for o in outputs]

                var_map = {k: [{x: y for x, y in o[k].items() if hasattr(
                    y, '__len__') or y is not None} if isinstance(o[k], dict) 
                    else o[k] for o in var if o[k] is not None] for k in var[0]}
                #var_map = {k: [x for x in v if len(x) > 0] for k, v in var_map.items()}
                var_map = {k: [x for x in v if not isinstance(x,dict) or len(x) > 0] 
                           for k, v in var_map.items()}

                # Restructure var_map as {'key' : [tensor,tensor,..]}
                _var_map = {}
                return_seq = self.graph.get_node_attributes('return_seq')[name]
                for k,v in var_map.items():
                    collect = {}
                    for i in v:
                        if isinstance(i,dict):
                            for m,n in i.items():
                                if m not in collect:
                                    collect[m] = []
                                collect[m].append(n)
                        else:
                            collect = tf.stack(v) if return_seq else v[-1] 

                    if isinstance(collect,dict):
                        for m,n in collect.items():
                            if return_seq:
                                collect[m] = tf.stack(n)
                            else:
                                collect[m] = v[-1]

                    _var_map[k] = collect

                return _var_map
            else:
                return self[name](output_dict or _X)

        traverse = cache(lambda name, depth=0: (name, (search(name, depth))))

        # can loop through and define
        for node in self.nodes:
            # if not hasattr(self[node], 'roll_out'):
            if not self[node].attributes['roll_out']:
                for input_node in nodes(node):
                    if self.graph.edges[input_node, node]['rollout_axis'] is not None:
                        rollout_axis = self.graph.edges[input_node,
                                                        node]['rollout_axis']
                        self[input_node].attributes['roll_out'] = Recurrence(
                            self, input_node, traverse, _X, rollout_axis)

        return self.find_and_select(flatten(dict(map(traverse, self.sinks))), 'output')

    # Should not have tensorflow in here but ...
    def test(self,seed=0):
        """ Generate a random input tensor and evaluate htg"""

        rng = np.random.default_rng(seed)
        X = {}
        for k,v in self.inputs.items():
            X[k] = rng.random(v.shape,dtype=v.dtype)

        return X,self(X)


    @property
    def summary(self):
        """ 
        Displays a summary of the graph and exchange
        of tensors.

        """
        # Check lazy build for subclassing
        self.build()

        def node_table(node):
            table = ColorTable([node.name, "Key", "Tensor Spec"],theme=Themes.GLARE_REDUCTION)

            if node.inputs:
                last = list(node.inputs.keys())[-1]
                for k,v in node.inputs.items():
                    table.add_row(['input',k,v],divider= k == last)

            if node.outputs:
                last = list(node.outputs.keys())[-1]
                for k,v in node.outputs.items():
                    table.add_row(['output',k,v],divider= k == last)

            if node is not self:
                edge_inputs = {}
                for e in list(self.graph.in_edges(node.name)):
                    s,t = e
                    X = self[s].outputs
                    if s == 'input':
                        X = self[t].inputs.copy()
                        if self[t].attributes['recurrent']:
                            for i in self[t].outputs:
                                if i in X:
                                    X.pop(i)

                    X = self._edge_mapping(X,s,t)
                    edge_inputs[(s,t)] = X

                for key,val in edge_inputs.items():
                    for k,v in val.items():
                        table.add_row([key,k,v],divider=False)
            print(table)

        # Parent
        node_table(self)

        # Children summary
        for k,v in self.nodes.items():
            if k not in ['input','output']:
                node_table(v)


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
        if not '.' in name:
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
        if source.count('.') == 1:
            ind = source.split('.', 1)[0]
            return node[ind]
        elif not '.' in source:
            return self

        # if parents is in a subgraph, recurse
        ind = source.split('.', 1)[0]
        source = source.split('.', 1)[1]

        return self._get_parent(source, node[ind])

    # def edge_label(self, source, target):
    #     """ Returns the edge label between source and target nodes

    #     Parameters:

    #     source : str
    #             The source node to get the output from

    #     target : str
    #             The target node to get the input from

    #     Returns: A dictionary with the edge label

    #     """

    #     self.logger.info(f'Getting edge label from {source} to {target} in HTG {self.name}')
    #     # get source and target nodes
    #     source_node = self._get_node(source)
    #     target_node = self._get_node(target)

    #     # get source outputs
    #     tmp_output = source_node.outputs if not source == 'input' else self.inputs
    #     tmp_output = self.get_outputs(self._get_parent(source)) if (
    #         '.' in source and not '.' in target) else tmp_output

    #     # convert source output format to list of labels
    #     if isinstance(tmp_output, dict):
    #         source_outputs = list(tmp_output.keys())
    #     else:
    #         source_outputs = tmp_output

    #     # get source output correspecting to the mapped output
    #     if len(source_node._outputs_map) > 0 or len(target_node._inputs_map) > 0:
    #         tmp_outputs = []
    #         for k in source_outputs:
    #             if k in source_node._outputs_map:
    #                 tmp_outputs.append(source_node._outputs_map[k])
    #             elif k in target_node._inputs_map:
    #                 tmp_outputs.append(target_node._inputs_map[k])
    #             else:
    #                 tmp_outputs.append(k)

    #         source_outputs = tmp_outputs

    #     # get target inputs
    #     tmp_input = target_node.inputs if not target == 'output' else self.outputs

    #     # convert target input format to list of labels
    #     if isinstance(tmp_input, dict):
    #         target_inputs = list(tmp_input.keys())
    #     else:
    #         target_inputs = tmp_input

    #     # get target input correspecting to the mapped input
    #     if len(source_node._outputs_map) > 0 and len(target_node._inputs_map) > 0:
    #         tmp_inputs = []
    #         for k in target_inputs:
    #             if k in target_node._inputs_map:
    #                 tmp_inputs.append(target_node._inputs_map[k])
    #             elif k in source_node._outputs_map:
    #                 tmp_inputs.append(source_node._outputs_map[k])
    #             else:
    #                 tmp_inputs.append(k)

    #         target_inputs = tmp_inputs

    #     # return dictionary of edge labels
    #     return {(source, target): (source_outputs, target_inputs)}

    # def get_all_edge_labels(self, graph):
    #     """ Returns all edge labels in the graph

    #     Parameters:

    #     graph : BaseGraph
    #             The graph to get the edge labels from

    #     Returns: A dictionary with the edge labels

    #     """

    #     self.logger.info(f'Getting all edge labels in HTG {self.name}')

    #     edge_labels = {}
    #     avial = {}
    #     req = {}

    #     # for all edges in the graph
    #     for edge in graph.edges:
    #         source = edge[0]
    #         target = edge[1]

    #         # get edge labels
    #         edge_labels.update(self.edge_label(source, target))

    #         # get available inputs
    #         if target in avial:
    #             if source == 'input':
    #                 avial[target] += edge_labels[(source, target)][1]
    #             else:
    #                 avial[target] += edge_labels[(source, target)][0]
    #         else:
    #             if source == 'input':
    #                 avial[target] = edge_labels[(source, target)][1]
    #             else:
    #                 avial[target] = edge_labels[(source, target)][0]
    #             if source == 'input':
    #                 avial[target] += edge_labels[(source, target)][1]
    #             else:
    #                 avial[target] += edge_labels[(source, target)][0]

    #         # get required inputs
    #         if target in req and not source == 'input':
    #             req[target] += edge_labels[(source, target)][1]
    #         else:
    #             req[target] = edge_labels[(source, target)][1]

    #     validity = {}
    #     # check if all required inputs are available
    #     for tar in avial.keys():
    #         avial[tar] = list(set(avial[tar]))
    #         req[tar] = list(set(req[tar]))

    #         # determine validity color based on matching between required and available inputs
    #         if all(x in avial[tar] for x in req[tar]):
    #             validity[tar] = True
    #         else:
    #             validity[tar] = False

    #     # colors = mcolors.CSS4_COLORS
    #     colors = mcolors.XKCD_COLORS
    #     # colors = mcolors.TABLEAU_COLORS

    #     edge_attr = {}
    #     for edge in graph.edges:
    #         source = edge[0]
    #         target = edge[1]

    #         random = rand.randint(1, len(list(colors.keys())))

    #         edge_attr[(source, target)] = {
    #             'validity': validity[target],
    #             'label': edge_labels[(source, target)],
    #             'source-color': list(colors.keys())[random]}

    #     return edge_attr

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

        # # get all edge labels and attributes
        # edge_labels = self.get_all_edge_labels(g)
        # label_output = []
        # for k, v in edge_labels.items():
        #     outputs = list(set(v['label'][0]))
        #     inputs = list(set(v['label'][1]))
        #     validity = v['validity']
        #     label_output.append(
        #         {'source': k[0], 'target': k[1], 'source output': outputs, 'target inputs': inputs, 'valid': validity})

        # convert to dataframe and apply color
        #label_output = pd.DataFrame(label_output)

        #label_output = label_output.style.apply(lambda x: [
        #    'background-color: green' if x['valid'] else 'background-color: red' for v in x], axis=1)

        # return dataframe
        #return label_output

    def draw(self, expand_nodes=None, layout='kamada_kawai_layout'):
        """ Draws the graph

        Parameters:

        expand_nodes : str, list, or 'all'

        layout : str

        Returns: A new graph with the expanded node.

        """

        self.logger.info("Drawing HTG %s", self.name)

        self.build()

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

        # # get all edge labels and attributes
        # edge_attributes = self.get_all_edge_labels(g)
        # edge_styles = ['solid' if v['validity'] else 'dashed' for k, v in edge_attributes.items()]
        # edge_colors = [v['source-color'] for k, v in edge_attributes.items()]

        # draw
        # nx.draw(g, pos, edge_color=edge_colors, style=edge_styles, with_labels=True, alpha=1, font_size=10, node_size=1000,
        #         node_color='white', font_color='darkblue', font_family='Impact')
        nx.draw(g, pos, with_labels=True, alpha=1, font_size=10, node_size=1000,
                node_color='white', font_color='darkblue', font_family='Impact')

class Recurrence:
    """
    Define recurrence for a given node in an HTG
    """

    @property
    def logger(self) -> logging.Logger:
        """ logger """
        return logging.getLogger(__name__)

    def __init__(self, node, name, traverse, X, rollout_axis):
        self.cache = {}
        self.node = node
        self.name = name
        self.traverse = traverse
        self.X = X
        self.rollout_axis = rollout_axis

        self.logger.info("Initializing Recurrence for node %s in HTG %s",name,node.name)

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
        is_recurrent = self.node.graph.get_node_attributes('recurrent')[self.name]
        if not is_recurrent:
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

        # Initialization of recurrent states
        if index == 0:
            self.cache[index] = self.node[self.name].initial_state(self.X)
            #self.cache[index] = self.node[self.name].attributes['roll_out'].initial_state

        elif index not in self.cache:
            input_values = {}

            # Index is not yet computed
            for input_node in dict(self.node.graph.in_edges(self.name)):

                # if the input node is recurrent
                if self.node.graph.edges[input_node, self.name]['rollout_axis'] is not None:
                    recurrence = self.node[input_node].attributes['roll_out']
                    input_values[input_node] = self.node._edge_mapping(recurrence[index - 1],input_node,self.name)
                else:

                    # get the value of the input node
                    _, output_value = self.traverse(input_node)
                    input_values[input_node] = output_value

            self.cache[index] = self.node[self.name](input_values)

        return self.cache[index]


    # @cached_property
    # def initial_state(self) -> dict:
    #     """ 
    #         Calls node initialization routine.
            
    #         Returns:
    #             Initial states
                
    #     """
    #     # Check that initialization has been defined
    #     if not self.node[self.name].attributes['initialization']:
    #         message = 'Recurrent node ({self.name} has not defined an initialzation routine. '
    #         message += 'recurrent nodes must define the initialzation routine.'
    #         raise ImproperTensorGraphError(message)

    #     return self.node[self.name].attributes['initialization'](self.X)

    # Removes 'input' or 'output' nesting of keys
    def flatten(self, d):
        """ Flatten input/output dictionaries """
        # find 'input' or 'output' items that are dicts
        keys = [k for k in ['input', 'output']
                if k in d and isinstance(d[k], dict)]

        # If found, flatten
        if keys:
            return [d.update(d.pop(k, {})) for k in ['input', 'output']] and d

        # otherwise, return original
        return d


class Identity(HierarchalTensorGraph):
    """ Idenity node used for I/O """
    registry_name = 'CREST_MODEL_HTG_IDENT'
    def __init__(self,name):
        super().__init__(name)
        self.node = lambda x : x

    def build(self):
        pass

    @property
    def is_basenode(self):
        return True
