from .BaseGraph import BaseGraph
import networkx as nx
import copy


class NetworkXGraph(BaseGraph):

    def __init__(self):
        self.graph = nx.DiGraph()

    @property
    def nodes(self):
        return self.graph.nodes

    def get_node_attributes(self,name):
        return nx.get_node_attributes(self.graph,name)

    def in_degree(self,node):
        return self.graph.in_degree(node)

    def out_degree(self,node):
        return self.graph.out_degree(node)

    def add_node(self,node,**attr):
        self.graph.add_node(node,**attr)
        
    def remove_node(self,node):
        self.graph.remove_node(node)

    def add_edge(self,from_node,to_node,**attr):
        self.graph.add_edge(from_node,to_node,**attr)
        
    def remove_edge(self,source, target):
        self.graph.remove_edge(source,target)
        
    def add_edges_from(self,ebunch_to_add, **attr):
        self.graph.add_edges_from(ebunch_to_add, **attr)
        
    def remove_edges_from(self,ebunch):
        self.graph.remove_edges_from(ebunch)

    def has_node(self,node):
        return self.graph.has_node(node)
    
    def has_edge(self,source,target):
        return self.graph.has_edge(source,target)
    
    def in_edges(self, nbunch=None, data=False, default=None):
        return self.graph.in_edges(nbunch,data,default)
    
    def out_edges(self, nbunch=None, data=False, default=None):
        return self.graph.out_edges(nbunch,data,default)
    
    def is_directed(self):
        return self.graph.is_directed()
    
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def edges(self):
        return self.graph.edges

    @property
    def is_empty(self):
        return nx.is_empty(self.graph)

    @property
    def is_directed_acyclic_graph(self):
         return nx.is_directed_acyclic_graph(self.graph)
    
    def is_isomorphic(self, other_graph):
        return nx.is_isomorphic(self.graph, other_graph)

    @property
    def adj(self):
        return self.graph.adj

    def __iter__(self):
        yield from self.graph
