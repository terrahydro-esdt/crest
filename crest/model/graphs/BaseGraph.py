from abc import ABC, abstractmethod


class BaseGraph(ABC):
    
    def __init__(self):
        self.graph = None
        
    @abstractmethod
    def nodes(self, node, **attr):
         raise NotImplementedError
            
    @abstractmethod
    def get_node_attributes(self,name):
        raise NotImplementedError

    @abstractmethod
    def in_degree(self,node):
        raise NotImplementedError

    @abstractmethod
    def out_degree(self,node):
        raise NotImplementedError

    @abstractmethod
    def add_node(self,node,**attr):
        raise NotImplementedError
        
    @abstractmethod
    def remove_node(self,node):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self,from_node,to_node,**attr):
        raise NotImplementedError
        
    @abstractmethod
    def remove_edge(self,source,target):
        raise NotImplementedError
        
    @abstractmethod
    def add_edges_from(self,ebunch,**attr):
        raise NotImplementedError
        
    @abstractmethod
    def remove_edges_from(self,ebunch):
        raise NotImplementedError
        
    @abstractmethod
    def has_node(self,node):
        raise NotImplementedError
        
    @abstractmethod
    def has_edge(self,node):
        raise NotImplementedError
    
    @abstractmethod
    def in_edges(self, nbunch=None, data=False, default=None):
        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self):
        raise NotImplementedError
        
    @property
    @abstractmethod
    def is_empty(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_directed_acyclic_graph(self):
         raise NotImplementedError
            
    @property
    @abstractmethod
    def is_empty(self):
         raise NotImplementedError

    @property
    @abstractmethod
    def adj(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError