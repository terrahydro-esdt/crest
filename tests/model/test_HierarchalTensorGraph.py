from math import exp

import pytest
import cloudpickle as pickle
import json
import os

from .helpers import *
from crest import HierarchalTensorGraph, ROOT_PATH
from crest.model.TensorGraph import ImproperTensorGraphError
from crest.model.LambdaNode import LambdaNode
from crest.model.TensorSpec import TensorSpec

root_path = os.path.join(ROOT_PATH.as_posix(),
                         os.path.join('..', 'tests', 'model'))


def test_basenode():
    m = HierarchalTensorGraph(
        node=lambda X: {'output': X['scalar'] * X['input']},
        name='mult'
    )
    assert m({'scalar': 2, 'input': 3}) == {'output': 6}

def test_add_node():
    # A multinode HierarchalTensorGraph
    mult = HierarchalTensorGraph(
        node=lambda X: {'mult': X['scalar'] * X['x']},
        name='mult'
    )

    # Adding a multinode HierarchalTensorGraph
    m = HierarchalTensorGraph(
        name='mult',
        inputs={'scalar': None, 'x': None},
        outputs={'mult': None}
    )
    m.add_edge('input', mult)
    m.add_edge(mult, 'output')
    assert m({'scalar': 2, 'x': 3}) == {'mult': 6}


def test_add_edges():
    # A multinode HierarchalTensorGraph
    mult = HierarchalTensorGraph(
        node=lambda X: {'mult': X['scalar'] * X['x']},
        name='mult'
    )

    # Adding a multinode HierarchalTensorGraph
    m = HierarchalTensorGraph(
        name='multiply',
        inputs={'scalar': None, 'x': None},
        outputs={'mult': None}
    )

    # Adding a HierarchalTensorGraph
    edges = [('input', mult), (mult, 'output')]
    m.add_edges_from(edges)
    assert m({'scalar': 2, 'x': 3}) == {'mult': 6}


def test_contains():
    m = AddMultExp()
    'add_mult' in m
    m['add_mult'] in m
    'exponentiator' in m
    m['exponentiator'] in m
    ('add_mult', 'adder') in m
    m[('add_mult', 'adder')] in m
    ('add_mult', 'multiplier') in m
    m[('add_mult', 'multiplier')] in m

    with pytest.raises(ImproperTensorGraphError):
        ['multiplier'] in m


def test_remove_nodes():
    def a(X):
        return {'a': 'a'}

    def b(X):
        return {'b': 'b'}

    m = HierarchalTensorGraph(
        name='test',
        inputs={'a': None, 'b': None},
        outputs={'a': None, 'b': None}
    )
    edges = [('input', a), ('input', b), (a, 'output'), (b, 'output')]
    m.add_edges_from(edges)
    assert m({'a': 'a', 'b': 'b'}) == {'a': 'a', 'b': 'b'}

    m.remove_node(b)
    m.outputs = {'a': None}
    assert m({'a': 'a', 'b': 'b'}) == {'a': 'a'}

    m = HierarchalTensorGraph(
        name='test',
        inputs={'a': None, 'b': None},
        outputs={'a': None, 'b': None}
    )
    edges = [('input', a), ('input', b), (a, 'output'), (b, 'output')]
    m.add_edges_from(edges)

    m.remove_node('b')
    m.outputs = {'a': None}
    assert m({'a': 'a', 'b': 'b'}) == {'a': 'a'}

    ha = HierarchalTensorGraph(a)
    hb = HierarchalTensorGraph(b)

    m = HierarchalTensorGraph(name='test')
    edges = [('input', ha), ('input', hb), (ha, 'output'), (hb, 'output')]
    m.add_edges_from(edges)

    m.remove_node(hb)
    m.outputs = {'a': None}
    assert m({'a': 'a', 'b': 'b'}) == {'a': 'a'}


def test_simple_coupled_model():
    m = AddMult()
    X = {'x_1': 0.2, 'x_2': 0.3, 'scalar': 1.5}
    res = m(X)
    assert res['add_mult_res'] == 0.75

    n = AddMultExp()
    res = n(X)
    assert pytest.approx(res['add_mult_exp_res']) == exp(0.75)


def test_unamed_model():
    with pytest.raises(ImproperTensorGraphError):
        m = HierarchalTensorGraph()


def test_uncallabe_node_basemodel():
    with pytest.raises(ImproperTensorGraphError):
        m = HierarchalTensorGraph('a', 'b')


def test_same_name_different_model():
    def fa(s):
        print('fa')

    def fb(s):
        print('fb')

    with pytest.raises(ImproperTensorGraphError):
        m1 = HierarchalTensorGraph(fa, 'a')
        m2 = HierarchalTensorGraph(fb, 'a')
        m3 = HierarchalTensorGraph(name='b')
        m3.add_edge(m1, m2)


def test_duplicate_models():
    def fa(s):
        print('fa')

    def fb(s):
        print('fb')

    with pytest.raises(ImproperTensorGraphError):
        m1 = HierarchalTensorGraph(fa, 'a')
        m2 = HierarchalTensorGraph(fb, 'b')
        m3 = HierarchalTensorGraph(name='c')
        m3.add_edge(m1, m2)
        m4 = HierarchalTensorGraph(name='d')
        m4.add_edge(m1, m3)


def test_add_edge_to_basemodel():
    def fa(s):
        print('fa')

    def fb(s):
        print('fb')

    with pytest.raises(ImproperTensorGraphError):
        m = HierarchalTensorGraph(lambda x: x, 'a')
        m.add_edge(fa, fb)


def test_cyclic_model():
    def fa(s):
        print('fa')

    def fb(s):
        print('fb')

    with pytest.raises(ImproperTensorGraphError):
        m = HierarchalTensorGraph('a')
        m.add_edge(fa, fb)
        m.add_edge(fb, fa)


def test_input_node_exist():
    m = HierarchalTensorGraph(name="m")
    m.add_edge('input', lambda x: x)
    with pytest.raises(ImproperTensorGraphError):
        m({'x': 1})


def test_input_is_not_target():
    m = HierarchalTensorGraph(name="m")
    with pytest.raises(ImproperTensorGraphError):
        m.add_edge(lambda x: x, 'input')


def test_output_is_not_source():
    m = HierarchalTensorGraph(name="m")
    with pytest.raises(ImproperTensorGraphError):
        m.add_edge('output', lambda x: x)


def test_no_key_found():
    def f1(x):
        return x
    m = HierarchalTensorGraph(
        name="m",
        inputs={'a': None}
    )
    m.add_edge('input', f1)
    m.add_edge('f1', 'output')
    with pytest.raises(ImproperTensorGraphError):
        m({'b': 1})


def test_multiple_keys_found():

    m1 = HierarchalTensorGraph(
        node=lambda X: {'f1': X['x']},
        name="m1",
        inputs={'x': None},
        outputs={'f1': None}
    )
    m2 = HierarchalTensorGraph(
        node=lambda X: {'f1': X['x']},
        name="m2",
        inputs={'x': None},
        outputs={'f1': None}
    )

    n = HierarchalTensorGraph(
        name="n",
        inputs={'x': None},
        outputs={'f1': None}
    )
    n.add_edge('input', m1)
    n.add_edge('input', m2)
    n.add_edge(m1, 'output')
    n.add_edge(m2, 'output')
    with pytest.raises(ImproperTensorGraphError):
        n({'x': 1})


def test_hanging_source_nodes():
    def a(x): return x['i'] + 1
    def b(x): return x['i'] * (x['a'] + 2)
    def c(x): return x['a'] + x['b']

    model = HierarchalTensorGraph(name='test')
    model.add_edge(a, b)
    # model.add_edge('input', 'a')
    model.add_edge('input', 'b')
    model.add_edge('a', c)
    model.add_edge('b', 'c')
    model.add_edge('c', 'output')
    with pytest.raises(ImproperTensorGraphError):
        model({'i': 2})


def test_hanging_sinks_nodes():
    def a(x): return x['i'] + 1
    def b(x): return x['i'] * (x['a'] + 2)
    def c(x): return x['b']

    model = HierarchalTensorGraph(name='test')
    model.add_edge(a, b)
    model.add_edge('input', 'a')
    model.add_edge('input', 'b')
    model.add_edge('b', c)
    model.add_edge('b', 'output')
    with pytest.raises(ImproperTensorGraphError):
        model({'i': 2})

def test_node_as_basenode():
    m = HierarchalTensorGraph(
        node=lambda X: {'output': X['scalar'] * X['input']},
        name='mult'
    )
    with pytest.raises(ImproperTensorGraphError):
         HierarchalTensorGraph(
                 node=m,
                 name='mult'
                 )

def test_pickling():
    X = {'x_1': 0.2, 'x_2': 0.3, 'scalar': 1.5}
    m = AddMultExp()
    res = m(X)

    n = pickle.loads(pickle.dumps(m))
    pickled_res = n(X)
    assert m(X) == n(X)


def compare_htg(node_1, node_2):
    inps = node_1.inputs.keys() == node_2.inputs.keys()

    for k in node_1.inputs.keys():
        inps = (TensorSpec(node_1.inputs[k]).spec_dict == TensorSpec(node_2.inputs[k]).spec_dict) and inps

    if (not inps):
        print(node_1.inputs.keys(), node_2.inputs.keys())
        print(TensorSpec(node_1.inputs['x']).spec_dict, node_2.inputs['x'].spec_dict)
    
    outs = node_1.outputs.keys() == node_2.outputs.keys()

    for k in node_1.outputs.keys():
        outs = (TensorSpec(node_1.outputs[k]).spec_dict == TensorSpec(node_2.outputs[k]).spec_dict) and outs

    if (not outs):
        print(TensorSpec(node_1.outputs['y']).spec_dict, node_2.outputs['y'].spec_dict)

    edge = node_1.edges == node_2.edges

    if (not edge):
        print(node_1.edges, node_2.edges)

    inpmap = node_1._inputs_map == node_2._inputs_map

    if (not inpmap):
        print(node_1._inputs_map, node_2._inputs_map)

    outmap = node_1._outputs_map == node_2._outputs_map

    if (not outmap):
        print(node_1._outputs_map, node_2._outputs_map)

    return inps and outs and edge and inpmap and outmap


# Test htg basenode that cannot be serialized
def test_json_basic_exception():
    def add(x): return x['a'] + x['b']

    with pytest.raises(Exception):
        htg_1 = HierarchalTensorGraph(node=add, name='add')
        htg_1.to_json()

# Test htg lambdabase node serialized locally and attempted to deserialize via HTG
def test_basenode_exception():
    l = LambdaNode(lambda X: X['x'] + 10, name='add', inputs={'x': None}, outputs={'add': None})

    assert HierarchalTensorGraph.from_json(l.to_json())

def test_json_basic():
    def add(x): return {'add': x['a'] + x['b']}
    l = LambdaNode(add, name='add', inputs={'a': None, 'b': None}, outputs={'add': None})

    htg_1 = HierarchalTensorGraph(name='add')
    htg_1.add_edge('input', l)
    htg_1.add_edge(l, 'output')

    htg_1.inputs = {'a': None, 'b': None}
    htg_1.outputs = {'add': None}

    json_data = htg_1.to_json()
    assert (not json_data == None)
    assert (isinstance(json_data, str))

    htg_2 = HierarchalTensorGraph.from_json(json_data)
    assert (not htg_2 == None)

    assert (htg_1(
        {'a': 1, 'b': 2}) == htg_2({'a': 1, 'b': 2}))

def test_json_model_exception():
    htg_1 = AddSequentialLayer()

    with pytest.raises(Exception):
        htg_1.to_json()


def test_json_model():
    keras_node = AddKerasModel()

    htg_1 = HierarchalTensorGraph(name='test_json_model')
    htg_1.add_edge('input', keras_node)
    htg_1.add_edge(keras_node, 'output')

    htg_1.inputs = {'x': None}
    htg_1.outputs = {'y': None}

    json_data = htg_1.to_json()
    
    htg_2 = HierarchalTensorGraph.from_json(json_data)
    assert (not htg_2 == None)

    assert (htg_1.edges == htg_2.edges)
    assert (htg_1.nodes['test'].node.get_config() ==
            htg_2.nodes['test'].node.get_config())


def test_json_double():

    htg_1 = AddSquareNode()

    json_data = htg_1.to_json()

    htg_2 = HierarchalTensorGraph.from_json(json_data)
    assert (not htg_2 == None)

    assert (htg_1.get_node('add') and htg_2.get_node('add'))
    assert (htg_1.get_node('square') and htg_2.get_node('square'))
    assert (htg_1.get_node('input') and htg_2.get_node('input'))
    assert (htg_1.get_node('output') and htg_2.get_node('output'))
    assert (htg_1.edges == htg_2.edges)

def test_json_triple():
    htg_1 = LogAddSquareNode()
    
    json_data = htg_1.to_json()

    htg_2 = HierarchalTensorGraph.from_json(json_data)
    assert (not htg_2 == None)

    assert (htg_1[('add_square', 'add')] and htg_2[('add_square', 'add')])
    assert (htg_1[('add_square', 'square')]
            and htg_2[('add_square', 'square')])
    assert (htg_1.get_node('log') and htg_2.get_node('log'))
    assert (htg_1.get_node('add_square') and htg_2.get_node('add_square'))
    assert (htg_1.get_node('input') and htg_2.get_node('input'))
    assert (htg_1.get_node('output') and htg_2.get_node('output'))
    assert (htg_1.edges == htg_2.edges)

    with pytest.raises(Exception) as ImproperTensorGraphError:
        htg_1.get_node('add')

    with pytest.raises(Exception) as ImproperTensorGraphError:
        htg_2.get_node('add')

    with pytest.raises(Exception) as ImproperTensorGraphError:
        htg_1.get_node('square')

    with pytest.raises(Exception) as ImproperTensorGraphError:
        htg_2.get_node('square')




def test_sqjson_custom():

    base_node = SquareRoot()
    htg = HierarchalTensorGraph(
        name='square_root'
    )

    htg.add_edge('input', base_node)
    htg.add_edge(base_node, 'output')

    htg.inputs = {'x': None}
    htg.outputs = {'square_root': None}

    json_data = htg.to_json()

    print(json_data)

    HierarchalTensorGraph.from_json(json_data)

def test_logjson_custom():

    htg = HierarchalTensorGraph(
        name='log',
        node=LogCustom()
    )

    json_data = htg.to_json()

    HierarchalTensorGraph.from_json(json_data)
