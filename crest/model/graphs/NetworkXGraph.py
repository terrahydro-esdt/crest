""" 
This module implements a NetworkX graph
object with only the required functionality.
"""

from .BaseGraph import BaseGraph
import networkx as nx
import copy


class NetworkXGraph(BaseGraph):
    """
    NetworkXGraph is a wrapper around the NetworkX DiGraph class.
    It is a subclass of BaseGraph and implements all of the methods
    defined in BaseGraph. Basic graph operations are implemented
    here. This is the graph utility of the HierarchalTensorGraph. 
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    @property
    def nodes(self):
        """ All nodes in the graph """
        return self.graph.nodes

    def get_node_attributes(self,name):
        """ Node attributes """
        return nx.get_node_attributes(self.graph,name)

    def get_edge_attributes(self,name):
        """ Edge attributes """
        return nx.get_edge_attributes(self.graph,name)

    def in_degree(self,node):
        """ Number of edges pointing to the node """
        return self.graph.in_degree(node)

    def out_degree(self,node):
        """ Number of edges pointing out of the node """
        return self.graph.out_degree(node)

    def add_node(self,node,**attr):
        """ Adds a node """
        self.graph.add_node(node,**attr)
        
    def remove_node(self,node):
        """ Removes a node """
        self.graph.remove_node(node)

    def add_edge(self,from_node,to_node,**attr):
        """ Add an edge """
        self.graph.add_edge(from_node,to_node,**attr)
        
    def remove_edge(self,source, target):
        """ Remove an edge """
        self.graph.remove_edge(source,target)
        
    def add_edges_from(self,ebunch_to_add, **attr):
        """ Add multiple edges """
        self.graph.add_edges_from(ebunch_to_add, **attr)
        
    def remove_edges_from(self,ebunch):
        """ Remove multiple edges """
        self.graph.remove_edges_from(ebunch)

    def has_node(self,node):
        """ If node is in graph """
        return self.graph.has_node(node)
    
    def has_edge(self,source,target):
        """ If edge is in graph """
        return self.graph.has_edge(source,target)
    
    def in_edges(self, nbunch=None, data=False, default=None):
        """ All edges coming in to a node """
        return self.graph.in_edges(nbunch,data=data,default=default)
    
    def out_edges(self, nbunch=None, data=False, default=None):
        """ All edges coming out of a node """
        return self.graph.out_edges(nbunch,data=data,default=default)
    
    def set_edge_attributes(self, attr: dict):
        """ Set edge attributes """
        nx.set_edge_attributes(self.graph, attr)
    
    def is_directed(self):
        """ True if directed graph """
        return self.graph.is_directed()
    
    def copy(self):
        """ Deep copy """
        return copy.deepcopy(self)
    
    @property
    def edges(self):
        """ All edges and attributes """
        return self.graph.edges

    @property
    def is_empty(self):
        """ True if the graph is empty """
        return nx.is_empty(self.graph)

    @property
    def is_directed_acyclic_graph(self):
        """ True if the graph is directive acyclic """
        return nx.is_directed_acyclic_graph(self.graph)
    
    def is_isomorphic(self, other_graph):
        """ True if the graph is isomorphic """
        return nx.is_isomorphic(self.graph, other_graph)

    @property
    def adj(self):
        """ Adjacency matrix """
        return self.graph.adj

    def __iter__(self):
        """ Iterate through graph """
        yield from self.graph
