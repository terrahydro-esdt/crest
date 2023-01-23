from functools import cache
import matplotlib.pyplot as plt
import networkx as nx

from .graphs import NetworkXGraph


class ImproperModelError(Exception):
    """ Raised when an improper model is created """
    pass


class Model:
    """

    CREST Hierarchal Model (CHM):

    This class defines the routines for creating models in CREST. A CHM
    has a unique name, a graph describing all the models and connections
    within the hierarchial model.

    Parameters
    ___________

    node : callable, optional
       A callable function to initalize a CHM. If set, this is not added to the graph; The graph remains
       empty indicating that this is a base model.

    name : str
       A UNIQUE name identifying the model. All models must be named and must be unique. If a 'node' parameter
       is passed that __contains__ a name, it will be used as the name of the model.
       If not, a name must be provided or an exception will be thrown.

    graph : BaseGraph
       A Diagonal Acylic Graph which specifies the connections and (sub)models within the CHM.

    """

    def __init__(self, node=None, name=None):
        self.name  = name or Model.get_name(node)
        self.node  = node or self
        self.graph = NetworkXGraph()

        # Check that the model has a name
        if self.name == 'None':
            message = f'A model name must be given, but a {self.name} was found'
            raise ImproperModelError(message)

        # Check that if node is passed it is callable
        if (self.node is not self) and (not callable(self.node)):
            message = f'node must be callable'
            raise ImproperModelError(message)



    def __repr__(self):
        """ Model instance representation """
        return f'Model("{self.name}", id={id(self)})'



    def __iter__(self):
        """ Iterate through all models within the hierarchal model """

        def traverse_models(graph,path):
            """ Recursive generator """
            for node in graph:
                yield [path + (graph.nodes[node]['model'].name,), graph.nodes[node]['model']]
                if not graph.nodes[node]['model'].is_empty:
                    yield from traverse_models(graph.nodes[node]['model'].graph,path + (graph.nodes[node]['model'].name,))

        yield from traverse_models(self.graph,(self.name,))



    def __getitem__(self, path: (str, tuple)) -> 'Model':
        """Retrieve the model which has the given path in the graph.
        
        Parameters
        ----------
        path : str or tuple
            If a string is given, returns the child model with the requested
            name. If a path is given, recursively calls __getitem__ to return
            the model with the given path relative to this graph.
             
        Returns
        -------
        Model
            Returns the Model at the requested path / with the given name.

        """
        # Base case, path is empty 
        if not path: return self

        # Recursively traverse path to find the requested Model
        if isinstance(path, str):
            if path == self.name: return self
            if path in ['input', 'output'] and not self.graph.has_node(path):
                return Model.identity(path)
            return self.graph.nodes[path]['model']
        return self[path[0]][path[1:]]



    def __contains__(self, path: (str, tuple)) -> bool:
        """Check if a Model with the given name / path exists in the graph.
        
        Parameters
        ----------
        path : str or tuple
            In the case of a string, the name of the child model within the
            parent. In the case of a path, the model with the given path 
            relative to the parent.

        Returns
        -------
        bool
            Whether or not the name or path exists within the graph.
        
        """
        if isinstance(path, str):
            return ((path == self.name) or
                    (path in ['input', 'output']) or 
                    self.graph.has_node(path))
        return path in self.all_models



    def __call__(self, X: dict) -> dict:
        """Propagate the given input through the graph.
        
        Parameters
        ----------
        X : dict
            Input of the model.

        Returns
        -------
        dict
            Dictionary containing outputs of the constructed
            graph, using the format {path: output value}. 
        
        """
        # Flatten input/output keys in the dict
        flatten = lambda d: [d.update(d.pop(k, {})) 
                    for k in ['input', 'output']] and d

        # Base case: graph is empty
        if self.is_empty: return self.node( flatten(X) )

        # Recursive case: traverse graph in reverse, from output to input
        nodes  = lambda name: dict(self.graph.in_edges(name))  # All input nodes
        search = lambda name: dict(map(traverse, nodes(name))) # Traverse all inputs
        output = lambda name: self[name]( search(name) or X )  # Get output for a node

        traverse = cache( lambda name: (name, output(name)) )
        return flatten( dict(map(traverse, self.sinks)) )



    @staticmethod
    def get_name(obj) -> str:
        """Try to determine the name of the given object.
        
        Parameters
        ----------
        obj : callable, Model
            The object from which to get the name.
        
        Returns
        -------
        str
            String representing the name of the given object.

        """
        value = getattr(obj, 'obj', None)
        if value is None:
            for attr in ['name', '__name__', '__qualname__']:
                value = value or getattr(obj, attr, None)
            return value or str(obj)
        return Model.get_name(value)



    @staticmethod
    def identity(name: str = 'identity') -> 'Model':
        """Create a Model with the identity function as its callable.
        
        Parameters
        ----------
        name : str
            The name given to the created Model.
        
        Returns
        -------
        Model
            Model which has the identity function as its callable.

        """
        return Model(lambda x:x, name)



    @property
    def all_models(self):
        """ Returns a dict of all child models within the parent """
        return dict([i for i in self])
        


    @property
    def models(self):
        """ Returns a dict of models within the parent model """
        return self.graph.get_node_attributes('model')



    @property
    def sources(self):
        """ Nodes with no incoming edges """
        is_source = lambda node: self.graph.in_degree(node) == 0
        return list(filter(is_source, self.graph.nodes))



    @property
    def sinks(self):
        """ Nodes with no outgoing edges """
        is_sink = lambda node: self.graph.out_degree(node) == 0
        return list(filter(is_sink, self.graph.nodes))



    @property
    def is_empty(self):
        """ Check if graph is empty """
        return nx.is_empty(self.graph)



    def search(self, name: str, partial: bool = False) -> dict:
        """Returns a dict {path: model} where path (partially) matches name.
        
        Parameters
        ----------
        name    : str
            The name serached for in the path.
        partial : bool
            Whether to search for an exact or partial match.
            
        Returns
        -------
        dict
            Dictionary mapping path to model, for all paths matching `name`.

        """
        transform = lambda path: ''.join(path) if partial else path
        is_match  = lambda item: name in transform(item[0])
        return dict(filter(is_match, self.all_models.items()))



    def get_model(self, node: (callable, str, 'Model')) -> 'Model':
        """Wrap the given node in a Model, and add to our graph if necessary.
        
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
        Model
            Model which represents the given node in the graph.

        Raises
        ------
        ImproperModelError
            - If a node with the same name already exists in the graph, but is
            not the same Model object.
            - If a duplicate model exists in the subgraph. (?)

        """

        # Fetch model by its name
        if isinstance(node, str):
            if node not in ['input', 'output']:
                assert(node in self), f'Unknown name "{node}"'
                return self[node]
            node = Model.identity(node)

        # If not a name (str) and not of type(Model) wrap it into a model
        if not isinstance(node, Model):
            node = Model(node)

            if node.name in self.graph:
                return self[node.name]

        # Check if you passed a different model with the same name
        if (node.name in self.graph) and (node is not self.models[node.name]):
            message = f'model ({node.name} = {node}) has the same name as existing model {self.models[node.name]}'
            raise ImproperModelError(message)

        # If model is not in the graph add it
        if not node.name in self.graph:
            
            # Models before adding
            before = self.all_models
            
            # Add model to graph
            self.graph.add_node(node.name, model=node)
            
            # Check we didn't duplicate a model that exist deeper in the graph
            seen = set()
            dupes = []
            for path, model in self.all_models.items():
                if model in seen:
                    dupes.append([path,model])
                else:
                    seen.add(model)
                    
            if dupes:
                match = [[path,model] for path,model in before.items() if model in dupes[0]]
                message = f'Duplicate models found {dupes[0][0]} and {match[0][0]}'
                raise ImproperModelError(message)
                
        return node



    def add_edge(self, 
        source: (callable, str, 'Model'), 
        target: (callable, str, 'Model'),
    ) -> None:
        """Add an edge between the source and target models.
        
        Parameters
        ----------
        source : str, callable, or Model
            The source node from which to begin the edge.
        target : str, callable, or Model
            The target node to draw an edge to.
        
        Raises
        ------
        ImproperModelError
            - If an edge is requested for a base Model (one which has a callable
            node attribute).
            - If an edge would create a cycle within the graph.

        """
        if self.node is not self:
            message = 'It is not permitted to add edges to a basemodel, '
            message+= 'i.e. a Model initialized with a callable node'
            raise ImproperModelError(message)
        
        source = self.get_model(source)
        target = self.get_model(target)
        
        if (source is not None) and (target is not None):
            self.graph.add_edge(source.name, target.name)
            if not self.graph.is_directed_acyclic_graph:
                message = f'Adding edge creates a cyclic graph'
                raise ImproperModelError(message)
                


    def draw(self, depth=None, _top=True):
        """Draw the model and internal graph.

        Parameters
        ----------
        depth : int, optional
            Depth with which to recursively expand sub-models when
            drawing the graph; e.g. depth=1 would only expand the 
            top level Model, and any sub-models would remain a single
            node in the drawn graph. When depth=None (the default), all
            sub-models are expanded recursively to the base Model.
        _top  : bool
            Private parameter used to signal execution is currently
            at the root Model (as opposed to a recursive call).

        """
        def create_io(nodes, io_name):
            io_nodes = set(nodes + [io_name])
            new_name = f'{self.name}.{io_name}'
            io_nodes.remove(io_name)
            io_nodes.add(new_name)
            return io_nodes, new_name

        # Generate the lists of input / output nodes
        inputs,  input_name  = create_io(self.sources, 'input' )
        outputs, output_name = create_io(self.sinks,   'output')


        def traverse(target, original_target=None):
            if target == output_name:
                original_target = 'output'

            in_edges = dict(self.graph.in_edges(original_target or target))
            edges    = set()

            # Traverse all incoming edges to the current target node
            for source in in_edges:
                source_model = self[source]

                if source == 'input':
                    source = input_name

                # Base Model
                if source_model.is_empty or (depth and depth-1) == 0:
                    edges.add((source, target))
                    edges = edges.union( traverse(source) )

                # Recursively expand Model
                else:
                    source_edges, source_inputs, source_outputs = \
                        source_model.draw(depth and depth-1, _top=False)

                    # Add source Model edges to current list
                    edges = edges.union(source_edges)

                    # Draw edges from all source outputs to the current target
                    for source_output in source_outputs:
                        edges.add((source_output, target))

                    # Use all source_inputs as the next target
                    for source_input in source_inputs:
                        edges = edges.union( traverse(source_input, source) )
            return edges


        # Traverse graph from output -> input, recursively expanding 
        # non-base Models as necessesary
        edges = set.union( *map(traverse, outputs) )

        # Add edge from input to all sources
        edges = edges.union((input_name, i) for i in inputs if i != input_name)

        # Add edge from all sinks to output
        edges = edges.union((o, output_name) for o in outputs if o != output_name)


        # Plot and show the graph if we're at the top level 
        if _top:
            graph = nx.DiGraph()
            graph.add_edges_from(edges)

            # pos = nx.spectral_layout(graph)
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=True, alpha=0.5)
            plt.show()

        return edges, [input_name], [output_name]
