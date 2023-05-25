from functools import cache
import networkx as nx
from types import MethodType
from .ExecutionGraph import ExecutionGraph
import matplotlib.pyplot as plt
from .graphs import NetworkXGraph
from typing import TypeVar,Union

class ImproperModelError(Exception):
    """ Raised when an improper model is created """
    pass

class HierarchalExecutionGraph(ExecutionGraph):
    """

    CREST Hierarchal Model (CHM):

    This class creates the hierarchal model structure of CREST.
    Its intention is to be inherited by other more specific models
    such as a CREST Tensor model.

    The CHM's main instance variable is a Directed Acyclic Graph (DAG).
    The DAG nodes are distinguished by their names type(str). A node in
    the DAG is itself a CHM and the reference to this model is contained
    in the the graph as the 'model' attribute. We use the term node to
    refer to models within a CHM.

    The term basemodel is used to distinguish a CHM which has an empty
    graph. Basemodels can be created from any callable function and act
    as the lowest layer in a CHM. By definition, basemodels cannot have
    any nodes or models within them. Therefore CHM models are built from
    either basemodels (callable functions) or other CHMs.

    A CHM itself is callable with a pre-defined __call__ that automatically
    executes all the nodes within the CHM as specified by its graph.

    With regard to names, all nodes or direct children of a parent CHM must
    have unique names. However, they can be the same as the name of nodes within
    its children. Within a CHM children are specified simply by their
    names. Nodes deeper in the model are specified by their 'path' where the path
    specifies the path to the CHM where the node is. Paths are specified as tuples.
    If for example, the parent is CHM A and contains nodes 'a1' and 'a2'. They
    can be accessed from A as A['a1'] and A['a2']. If within a1 there exist nodes
    b1 and b2, they can be accessed from A as A[('a1','b1')] and A[('a1','b2')] or
    accessed from 'a1' as a1['b1'] and a1['b2'].

    With regard to input and output nodes. Every CHM must have an input and output
    (except basemodels). I/O nodes are added to the CHM by specifying at least one
    edge from 'input' to a model and one edge from a model to 'output'. CHM also
    assume all I/O of models are dictionaries. Similarly input and output deeper
    within a hierarchal model are specified by their path if one needs
    to access them. In addition, CHM require specification of the input and output
    keys by setting input_features and output_features instance variables type(list).
    Generally, simply a list of key strings is sufficient. In cases where, inputs
    to a model may come from multiple models which may have the same keys a tuple
    specifying which input to use will be required. The framework will throw an
    exception and list possible tuple options.


    Parameters
    ----------

    node : callable, optional
       A callable function to initalize a CHM. If set, this is not added to the graph;
       The graph remains empty and a basemodel is created.

    name : str
       A UNIQUE name identifying the model. All models must be named and must be unique.
       If a 'node' parameter is passed that __contains__ a name,
       it will be used as the name of the model.
       If not, a name must be provided or an exception will be thrown.

    graph : BaseGraph
       A Diagonal Acylic Graph which specifies the connections and nodes within the CHM.

    input_features : list
       A list of input features to be selected. List items can be a
       string, tuple, or key-value pair.

    output_features : list
       A list of input features to be selected. List items can be a
       string, tuple, or key-value pair.


    """

    def __init__(self, node: callable=None, name: str=None):

        self.name  = name or HierarchalExecutionGraph.get_name(node)
        self.node  = node  or self
        self.graph  = NetworkXGraph()
        self.output = []
        self.input_features = {}
        self.output_features = {}

        #check that the model has a name
        if self.name == 'None':
            message = f'A model name must be given, but a {self.name} was found'
            raise ImproperModelError(message)

        #check that if node is passed it is callable
        if (self.node is not self) and (not callable(self.node)):
            message = f'node must be callable'
            raise ImproperModelError(message)

    def __iter__(self):
        """ Iterate through all models within the hierarchal model """

        def traverse_models(graph,path):
            """ Recursive generator """
            for node in graph:
                yield [path + (graph.nodes[node]['model'].name,), graph.nodes[node]['model']]
                if not graph.nodes[node]['model'].is_empty:
                    yield from traverse_models(graph.nodes[node]['model'].graph,path + (graph.nodes[node]['model'].name,))

        yield from traverse_models(self.graph,())

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self,d):
        self.__dict__ = d

    @staticmethod
    def get_name(obj) -> str:
        """

        Try to determine the name of the given object
        using in order 'name', '__name__', and '__qualname__'.

        Parameters
        ----------
        obj : callable, Model
             The object from which to get the name

        """

        value = getattr(obj, 'obj', None)
        if value is None:
            for attr in ['name', '__name__', '__qualname__']:
                value = value or getattr(obj, attr, None)
            return value or str(obj)
        return HierarchalExecutionGraph.get_name(value)

    def search(self, name: str, partial: bool=False) -> dict:
        """

        Tries to find the model specified with the given name.
        Returns match/matches as dict with key = tuple and
        value = Model.

        Parameters
        ----------
        name : string or tuple
             The name serached for in the path

        partial : bool
             Whether to search for an exact or partial match. If tuple is given,
             only exact match will be used and partial will be ignored.


        """
        #tuples must be exact matches.
        if isinstance(name,tuple):
            return dict(filter(lambda t: name == t[0], self.all_models.items()))

        if(partial):
            return dict(filter(lambda s : name in ''.join(s[0]), self.all_models.items()))
        return dict(filter(lambda s : name in s[0], self.all_models.items()))

    @property
    def all_models(self) -> dict:
        """ Returns a dict of all child models within the parent """
        return dict([i for i in self])

    @property
    def edges(self):
        return self.graph.edges

    @property
    def models(self) -> dict:
        """ Returns a dict of models within the parent model """
        return self.graph.get_node_attributes('model')

    @property
    def sources(self) -> list:
        """ Nodes with no incoming edges """
        is_source = lambda node: self.graph.in_degree(node) == 0
        return list(filter(is_source,self.graph.nodes))


    @property
    def sinks(self) -> list:
        """ Nodes with no outgoing edges """
        is_sink = lambda node: self.graph.out_degree(node) == 0
        return list(filter(is_sink,self.graph.nodes))


    @property
    def is_empty(self) -> bool:
        """ Check if graph is empty """
        return self.graph.is_empty

    @staticmethod
    def identity(name: str) -> 'HierarchalExecutionGraph':
        """
        Create an identity Model.

        name : string
             The name given to the identity Model created

        """
        return HierarchalExecutionGraph(lambda x:x,name)

    def get_model(self, node: Union[str,callable]) -> 'HierarchalExecutionGraph':
        """
        Wrap the given node in a Model, and add to our graph if necessary.

        Parameters
        ----------
        node : str, callable, or Model
            - If a callable is passed, it is wrapped into a Model and added to
            the graph if it doesn't yet exist.
            - If a string is passed, the Model (which must already exist in
            the graph) is returned.
            - If a Model is passed, it is added to the graph if it doesn't yet
            exist.

        Returns
        -------
        HierarchalExecutionGraph
            Model which represents the given node in the graph.

        Raises
        ------
        ImproperModelError
            - If a node with the same name already exists in the graph, but is
            not the same Model object.
            - If a the same referenced model passed already exists in hierarchal model.
            - If node is not callable, a string, or a HierarchalExecutionGraph

        """
        if not (isinstance(node,str) or isinstance(node,HierarchalExecutionGraph) or callable(node)):
            message = f'node must be either callable, a string, or a HierarchalExecutionGraph'
            raise ImproperModelError(message)

        #fetch model by its name
        if isinstance(node, str):
            #add io node if first time called
            if (node in ['input','output']) and (node not in self):
                io =  HierarchalExecutionGraph.identity(node)
                self.graph.add_node(io.name, model=io)
                return io

            #otherwise if using a str() it must already be in the graph
            assert(node in self), f'Unknown name "{node}"'
            return self[node]

        #if not a name (str) and not of type(Model)
        if not isinstance(node, HierarchalExecutionGraph):

            #check if basemodel with this name exist and if it has the same node value.
            #if it has the same node value then this refers to that basemodel.
            name = HierarchalExecutionGraph.get_name(node)
            if (name in self.graph) and (node is self[name].node):
                    return self[name]

            #if it doesn't exist we need to create a basemodel
            node = HierarchalExecutionGraph(node)

        #check if you passed a different model with the same name
        if (node.name in self.graph) and (node is not self.models[node.name]):
            message = f'model ({node.name} = {node}) has the same name'
            message += f'(and path) as existing model {self.models[node.name]}'
            raise ImproperModelError(message)

        #if model is not in the graph add it
        if not node.name in self.graph:

            #models before adding
            before = self.all_models

            #add model to graph
            self.graph.add_node(node.name, model=node)

            #check we didn't duplicate a model (same id) that exist deeper in the graph
            seen = set()
            dupes = []
            for path,model in self.all_models.items():
                if model in seen:
                    dupes.append([path,model])
                else:
                    seen.add(model)

            if dupes:
                match = [[path,model] for path,model in before.items() if model in dupes[0]]
                message = f'Duplicate models found {dupes[0][0]} and {match[0][0]}'
                raise ImproperModelError(message)

        return node

    def add_node(self, node: Union[callable,'HierarchalGraphmodel']):
        """
        Add a node to the HierarchalExecutionGraph.

        Parameters
        ----------
        node : callable, or HierarchalExecutionGraph
            - If a callable is passed, it is wrapped into a Model and added to
            the graph if it doesn't yet exist.
            - If a Model is passed, it is added to the graph if it doesn't yet
            exist.

        Raises
        -------

        ImproperModelError
            -if not callable or a HierarchalExecutionGraph
        """

        if not (callable(node) or  isinstance(node,HierarchalExecutionGraph)):
            message = f'A node must be either callable or a HierarchalExecutionGraph'
            raise ImproperModelError(message)

        #we'll use get_model to add it to the graph. Get model will check if exist
        #and only add it if does not. It also checks node meets various criteria before
        #adding it to the graph.
        self.get_model(node)

    def remove_node(self,node):
        """
        Removes a node from the HierarchalExecutionGraph

        """

        #remove node from string
        if isinstance(node,str):
            if node in self.models:
                self.graph.remove_node(node)
                return
            else:
                message=f'Node with name {node} not found'
                raise ImproperModelError(message)

        #remove node from HierarchalExecutionGraph
        if isinstance(node,HierarchalExecutionGraph):
            if not node.name in self.models:
                message=f'Node with name {node} not found'
                raise ImproperModelError(message)

            if not self[node.name] is node:
                message=f'Node with the name {node.name} exist but does match the one passed'
                raise ImproperModelError(message)

            self.graph.remove_node(node.name)
            return

        #remove node from basemodel function
        #check if basemodel with this name exist and if it has the same node value.
        #if it has the same node value then this refers to that basemodel.
        if callable(node):
            name = HierarchalExecutionGraph.get_name(node)
            if not name in self.graph:
                message=f'Node with name {node} not found'
                raise ImproperModelError(message)

            if not node is self[name].node:
                message=f'Node with the name {node.name} exist but does match the one passed'
                raise ImproperModelError(message)

            self.graph.remove_node(name)
            return

        #if not one of the above, throw an exception
        message=f'Unrecognized node type. Must be of type str, callable, or HierarchalExecutionGraph'
        raise ImproperModelError(message)



    def add_edge(self, source: Union[str,callable], target: Union[str,callable], **attr):
        """

        Add an edge between the source and target models.

        Parameters
        ----------

        source : str, callable, Model
             The source from which to begin the edge

        target : str, callable, Model
             The sink from which to end the edge

        Raises
        ------
        ImproperModelError
            - If an edges is added to a basemodel
            - If 'input' is passed as a sink
            - If 'output' is passed as a source
            - If a cyclic graph is created by adding the edge

        """

        if (isinstance(source,str)):
            if source == 'output':
                message = f'Output cannot be used as a source'
                raise ImproperModelError(message)

        if(isinstance(target,str) & (self.node is self)):
            if target == 'input':
                message = f'Input cannot be used as a target'
                raise ImproperModelError(message)

        if (self.node is not self):
            message = f'It is not permitted to add edges to a basemodel,'
            message += 'i.e. a Model initialized with a callable node'
            raise ImproperModelError(message)

        source = self.get_model(source)
        target = self.get_model(target)


        if (source is not None) and (target is not None):
            self.graph.add_edge(source.name, target.name)
            if not self.graph.is_directed_acyclic_graph:
                message = f'Adding this edge created a cyclic graph'
                raise ImproperModelError(message)

    def add_edges_from(self,ebunch: list):
        """
        Add multiple edges

        Parameters
        ----------

        ebunch: a list of tuples (source,target)

        """

        for i in ebunch:
            self.add_edge(*i)

    def __repr__(self):
        return f'HierarchalExecutionGraph("{self.name}", id={id(self)}) '


    def __getitem__(self, path: Union[str,tuple]) -> 'HierarchalExecutionGraph':

        """ Retrieve the model which has the given name from our graph

        Parameters
        ----------
        path : string or tuple
             In the case of a string, the name of the child model within the parent.
             In the case, of a path, the model with the given path relative to the parent.

        Returns
        ----------

        Model with path = 'path'

        """

        if not path:
            return self
        if(isinstance(path,str)):
            #if path == self.name: return self
            #return i/o nodes
            if path in ['input'] and (path not in self.graph):
                return HierarchalExecutionGraph.input_node(path)

            if path in ['output'] and (path not in self.graph):
                return HierarchalExecutionGraph.output_node(path)

            return self.graph.nodes[path]['model']
        return self[path[0]][path[1:]]


    def __contains__(self, path: Union[str,tuple]) -> bool:
        """ Check if a model with the given name is in our graph.

        Parameters
        ----------
        path : string or tuple
              In the case of a string, the name of the child model within the parent.
              In the case of a path (tuple), the model with the given path relative
              to the parent.

        """
        #if given a string check if it is direct child of model
        if isinstance(path,str):
            return (path == self.name) or self.graph.has_node(path)

        return path in self.all_models

    def feature_map(self,X: dict, io: str) -> dict:
        """

        Attempts to select the input/output with keys specified in
        input/output_features.

        Parameters
        ----------

        X : Input of the model

        io: either 'input' or 'output'

        Raises
        ------

        ImproperModelError:
         - if the key is not found.
         - if multiple keys (coming from different models)
         of that name are found. In this case,
         the found keys will be given in the error message.

        """

        def flatten_dict(d,key):
            for k,v in d.items():
                if isinstance(v, dict):
                    yield from flatten_dict(v,key + tuple([k]))
                else:
                    yield [key + tuple([k]), v]

        def find(name,features):
            found = [i for i in features.keys() if name in i or name == i ]

            if not found:
                message = f'Could not find {io} key = {name} in '
                message += f'Model[{self.name}]. Options: {features}.'
                raise ImproperModelError(message)

            if len(found) > 1:
                print(features,name)
                message = f'Found multiple {io} keys = {found} for {name}. '
                message += f'Use one of these tuple in feature list.'
                raise ImproperModelError(message)

            return found[0]

        if io == 'input':
            features_list = self.input_features
        else:
            features_list = self.output_features

        if features_list:
            if not isinstance(features_list,list):
                features_list = [features_list]
            f = dict([i for i in flatten_dict(X,())])
            replace = {j : i[j] for i in features_list
                       if isinstance(i,dict) for j in i}
            get = lambda x : list(x.keys())[0] if isinstance(x,dict) else x
            features = [get(i) for i in features_list]
            features = {i : f[find(i,f)] for i in features}

            #make replacments
            for k,v in replace.items():
                features[v] = features.pop(k)
            return features

        return X

    def set_feature_map(self,f: callable):
        """

        Define the feature mapping function.

        f : The feature mapping to use on the input

        """

        return MethodType(f,self)

    def reset_feature_map(self):
        self.set_feature_map(lambda x,X : X)


    def __call__(self, X: dict) -> dict:
        """Propagate the given input through the graph.

        Parameters
        ----------

        X : dict
            Input of the model.

        Returns
        -------

        Dictionary containing outputs of the constructed
        graph, using the format {path: output value}.

        Raises
        ------

        ImproperModelError
            - If input and output nodes do not exists

        """

        #apply feature map
        _X = self.feature_map(X,'input')

        #if not a basemodel
        if(self.node is self):

            #check that i/o exist
            if not all([b in self.graph for b in ['input','output']]):
                message = f'A model name must contain i/o nodes'
                raise ImproperModelError(message)

            #check that only i/o is a source/sink
            sources = list(filter(lambda x : x != 'input',self.sources))
            if sources:
                message = f'Only input can be a source node in the graph. '
                message += f'Found sources {[ self[i] for i in sources]}'
                raise ImproperModelError(message)

            sinks = list(filter(lambda x : x != 'output',self.sinks))
            if sinks:
                message = f'Only output can be a sink in the graph. '
                message += f'Found sinks {[ self[i] for i in sinks]}'
                raise ImproperModelError(message)

        # Flatten input/output keys in the dict
        flatten = lambda d: [d.update(d.pop(k, {}))
                    for k in ['input', 'output']] and d

        # Base case: graph is empty
        if self.is_empty: return self.feature_map( self.node(flatten(_X)), 'output')

        # Recursive case: traverse graph in reverse, from output to input
        nodes  = lambda name: dict(self.graph.in_edges(name))  # All input nodes
        search = lambda name: dict(map(traverse, nodes(name))) # Traverse all inputs
        output = lambda name: self[name]( search(name) or _X )  # Get output for a node


        def output2(name):
            self[name].output = self[name]( search(name) or _X )
            return self[name].output

        traverse = cache( lambda name: (name, output2(name)) )
        self.output = self.feature_map( flatten(dict(map(traverse, self.sinks))), 'output')
        return  self.output


    def expand_graph_node(self,nodename: str,g=None):
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

        #if basenode cannot be expanded
        if self[nodename].is_empty:
            return graph

        #in/out edges of node
        in_node_edges =  graph.in_edges(nodename)
        out_node_edges =  graph.out_edges(nodename)

        #input/output edges of within the node
        edges_within_node = self[nodename].edges
        in_edges = [e for e in edges_within_node if 'input' in e]
        out_edges = [e for e in edges_within_node if 'output' in e]

        #create edges by fusing edges and adding parent to the names
        create_edges = [(x[0],nodename + '.' +  y[1]) for x in in_node_edges for y in in_edges]
        create_edges += [(nodename + '.' +  x[0],y[1]) for x in out_edges for y in out_node_edges]
        create_edges += [(nodename + '.' + x[0],nodename + '.' + x[1]) for x in
                         [e for e in edges_within_node if ('input' not in e) and (not 'output' in e)]]
        subnodes = {nodename + '.' +  k : node for (k,node) in self[nodename].models.items()
                    if k not in ['input','output']}

        #copy graph and remove node
        graph.remove_node(nodename)

        #add in new edges and subnodes
        for k,v in subnodes.items():
            graph.add_node(k,model=v)
        graph.add_edges_from(create_edges)

        return graph

    def draw(self,expand_nodes=None,layout='kamada_kawai_layout'):

        #check if submodels are empty
        def is_not_all_empty(graph):
            for v in graph:
                if not graph.nodes[v]['model'].is_empty:
                    return True
            return False

        g = self.graph

        if expand_nodes:
            if not isinstance(expand_nodes,list) and expand_nodes != 'all':
                for node in [expand_nodes]:
                    g = self.expand_graph_node(node,g)

            #Expand graph nodes and create new graph
            if isinstance(expand_nodes,list):
                for node in expand_nodes:
                    g = self.expand_graph_node(node,g)


            if expand_nodes == 'all':
                while(is_not_all_empty(g)):
                    models = g.get_node_attributes('model')
                    for k,node in models.items():
                        if not node.is_empty:
                            g = self.expand_graph_node(node.name,g)

        #Draw the final graph
        if layout == 'kamada_kawai_layout':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'spetral_layout':
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = None

        nx.draw(g,pos,with_labels=True,alpha=1,font_size=10,node_size=1000,
                node_color='white',font_color='darkblue',font_family='Impact')
