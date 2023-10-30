from crest.model.graphs.NetworkXGraph import NetworkXGraph as graph


def test_add_node():
    g = graph()
    g.add_node('a')
    assert g.has_node('a')
    
def test_remove_node():
    g = graph()
    g.add_node('a')
    g.add_node('b')
    g.add_edge('a','b')
    g.remove_node('b')
    
    assert not g.has_node('b') #no node
    assert not g.has_edge('a','b') #no edges
    
def test_add_edge():
    g = graph()
    g.add_node('a')
    g.add_node('b')
    g.add_edge('a','b')
    assert g.has_edge('a','b')
    
def test_remove_edge():
    g = graph()
    g.add_node('a')
    g.add_node('b')
    g.add_edge('a','b')
    g.remove_edge('a','b')
    assert not g.has_edge('a','b')
    
def test_add_edges_from():
    g = graph()
    g.add_node('a')
    g.add_node('b')
    edges = [('input','a'),('input','b'),('b','output'),('a','output')]
    g.add_edges_from(edges)
    for i in edges:
        assert g.has_edge(*i)
    
def test_remove_edges_from():
    g = graph()
    g.add_node('a')
    g.add_node('b')
    edges = [('input','a'),('input','b'),('b','output'),('a','output')]
    g.add_edges_from(edges)
    g.remove_edges_from(edges)
    assert not list(g.edges)