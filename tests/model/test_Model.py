import pytest
from .helpers import AddMult,AddMultExp
from ...src import HierarchalGraphModel,ImproperModelError
from math import exp


def test_basemodel():
    def mult(x): return x['scalar'] * x['x']
    m = HierarchalGraphModel(node=mult,name='mult')
    assert m({'scalar':2,'x':3}) == 6
    
def test_simple_coupled_model():
    m = AddMult()
    X = {'x_1' : 0.2, 'x_2' : 0.3,'scalar' : 1.5}
    res = m(X)
    assert res['res_add_mult'] == 0.75
    
    n = AddMultExp()
    res = n(X)
    assert pytest.approx(res['res_add_mult_exp']) == exp(0.75)

def test_unamed_model():
    with pytest.raises(ImproperModelError):
        m = HierarchalGraphModel()
        
def test_uncallabe_node_basemodel():
    with pytest.raises(ImproperModelError):
        m = HierarchalGraphModel('a','b')
        
def test_same_name_different_model(): 
    def fa(s):
        print('fa')
                  
    def fb(s):
        print('fb')
        
    with pytest.raises(ImproperModelError):
        m1 = HierarchalGraphModel(fa,'a')
        m2 = HierarchalGraphModel(fb,'a')
        m3 = HierarchalGraphModel(name='b')
        m3.add_edge(m1,m2)

def test_duplicate_models():
    def fa(s):
        print('fa')
                  
    def fb(s):
        print('fb')
                  
    with pytest.raises(ImproperModelError):
        m1 = HierarchalGraphModel(fa,'a')
        m2 = HierarchalGraphModel(fb,'b')
        m3 = HierarchalGraphModel(name='c')
        m3.add_edge(m1,m2)
        m4 = HierarchalGraphModel(name='d')
        m4.add_edge(m1,m3)        
                  
def test_add_edge_to_basemodel():
    def fa(s):
        print('fa')
                  
    def fb(s):
        print('fb')
                  
    with pytest.raises(ImproperModelError):
        m = HierarchalGraphModel(lambda x:x,'a')
        m.add_edge(fa,fb)
        
def test_cyclic_model():
    def fa(s):
        print('fa')
                  
    def fb(s):
        print('fb')
                  
    with pytest.raises(ImproperModelError):
        m = HierarchalGraphModel('a')
        m.add_edge(fa,fb)
        m.add_edge(fb,fa)
        
def test_input_node_exist():
    m = HierarchalGraphModel(name="m")
    m.add_edge('input',lambda x:x)
    with pytest.raises(ImproperModelError):
        m({'x': 1})
        
def test_input_is_not_target():
    m = HierarchalGraphModel(name="m")
    with pytest.raises(ImproperModelError):
        m.add_edge(lambda x:x, 'input')
        
def test_output_is_not_source():
    m = HierarchalGraphModel(name="m")
    with pytest.raises(ImproperModelError):
        m.add_edge('output',lambda x:x)
        
def test_no_key_found():
    def f1(x):
        return x
    m = HierarchalGraphModel(name="m")
    m.add_edge('input',f1)
    m.add_edge('f1','output')
    m.input_features = 'a'
    with pytest.raises(ImproperModelError):
        m({'b':1})
        
def test_multiple_keys_found():
    def f1(X):
        return {'f1' : X['x']}
    
    m = HierarchalGraphModel(name="m")
    m.add_edge('input',f1)
    m.add_edge('f1','output')

    n = HierarchalGraphModel(name="n")
    n.add_edge('input',m)
    n.add_edge('input',f1)
    n.add_edge('f1','output')
    n.output_features = 'f1'
    with pytest.raises(ImproperModelError):
        n({'x':1})

def test_hanging_source_nodes():
    def a(x): return x['i'] + 1 
    def b(x): return x['i'] * (x['a'] + 2)
    def c(x): return x['a'] + x['b']

    model = HierarchalGraphModel(name='test')
    model.add_edge(a, b)
    # model.add_edge('input', 'a')
    model.add_edge('input', 'b')
    model.add_edge('a', c)
    model.add_edge('b', 'c')
    model.add_edge('c', 'output')
    with pytest.raises(ImproperModelError):
        model({'i':2})
        
def test_hanging_sinks_nodes():        
    def a(x): return x['i'] + 1 
    def b(x): return x['i'] * (x['a'] + 2)
    def c(x): return x['b']

    model = HierarchalGraphModel(name='test')
    model.add_edge(a, b)
    model.add_edge('input', 'a')
    model.add_edge('input', 'b')
    model.add_edge('b', c)
    model.add_edge('b', 'output')
    with pytest.raises(ImproperModelError):
        model({'i':2})