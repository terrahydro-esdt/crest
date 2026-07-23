""" 
This module implements the base graph class 
and specifies the required functions 
"""

from abc import ABC, abstractmethod


class BaseGraph(ABC):
    """
    Base class for all graph types in CREST. 
    This class is an abstract class and should not be instantiated.
    """
    
    def __init__(self):
        self.graph = None
        
    @abstractmethod
    def nodes(self, node, **attr):
        """ All nodes in the graph """
        raise NotImplementedError
            
    @abstractmethod
    def get_node_attributes(self,name):
        """ Node attributes """
        raise NotImplementedError

    @abstractmethod
    def in_degree(self,node):
        """ Number of edges pointing to the node """
        raise NotImplementedError

    @abstractmethod
    def out_degree(self,node):
        """ Number of edges pointing out of the node """
        raise NotImplementedError

    @abstractmethod
    def add_node(self,node,**attr):
        """ Adds a node """
        raise NotImplementedError
        
    @abstractmethod
    def remove_node(self,node):
        """ Removes a node """
        raise NotImplementedError

    @abstractmethod
    def add_edge(self,from_node,to_node,**attr):
        """ Add an edge """
        raise NotImplementedError
        
    @abstractmethod
    def remove_edge(self,source,target):
        """ Remove an edge """
        raise NotImplementedError
        
    @abstractmethod
    def add_edges_from(self,ebunch,**attr):
        """ Add multiple edges """
        raise NotImplementedError
        
    @abstractmethod
    def remove_edges_from(self,ebunch):
        """ Remove multiple edges """
        raise NotImplementedError
        
    @abstractmethod
    def has_node(self,node):
        """ If node is in graph """
        raise NotImplementedError
        
    @abstractmethod
    def has_edge(self,node):
        """ If edge is in graph """
        raise NotImplementedError
    
    @abstractmethod
    def in_edges(self, nbunch=None, data=False, default=None):
        """ All edges coming in to a node """
        raise NotImplementedError
    
    def out_edges(self, nbunch=None, data=False, default=None):
        """ All edges coming out of a node """
        raise NotImplementedError

    def set_edge_attributes(self, attr: dict):
        """ Set edge attributes """
        raise NotImplementedError
    
    def is_directed(self):
        """ True if directed graph """
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self):
        """ All edges and attributes """
        raise NotImplementedError
        
    @property
    @abstractmethod
    def is_empty(self):
        """ True if the graph is empty """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_directed_acyclic_graph(self):
        """ True if the graph is directive acyclic """
        raise NotImplementedError
    
    def is_isomorphic(self, other_graph):
        """ True if the graph is isomorphic """
        raise NotImplementedError
            
    @property
    @abstractmethod
    def adj(self):
        """ Adjacency matrix """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        """ Iterate through graph """
        raise NotImplementedError
