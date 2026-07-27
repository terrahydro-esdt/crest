from math import exp

import pytest
import cloudpickle as pickle
import os
from .helpers import *
from crest import ROOT_PATH
from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
from crest.model.TensorGraph import ImproperTensorGraphError
from crest.model import TensorSpec
from crest.model.Node import Node

root_path = os.path.join(ROOT_PATH.as_posix(),
                         os.path.join('..', 'tests', 'model'))

def test_alias_long():
    m = Node(
        node=lambda X: {'output': X['scalar'] * X['input']},
        name='mult',
        inputs={},
        outputs={}
    )

    assert m({'scalar': 2, 'input': 3}) == {'output': 6}

def test_alias_short():
    m = Node(
        node=lambda X: {'output': X['scalar'] * X['input']},
        name='mult',
        inputs={},
        outputs={}
    )

    assert m({'scalar': 2, 'input': 3}) == {'output': 6}


def test_basenode():
    m = Multiplier()
    assert m({'scalar': 2, 'x': 3}) == {'product': 6}


def test_add_node():
    # A multinode HierarchalTensorGraph
    mult = Multiplier()

    # Adding a multinode HierarchalTensorGraph
    m = HierarchalTensorGraph(name='mult')
    m.add_edge('input', mult)
    m.add_edge(mult, 'output')
    assert m({'scalar': 2, 'x': 3}) == {'product': 6}

def test_add_edges():
    # A multinode HierarchalTensorGraph
    mult = Multiplier()
    m = HierarchalTensorGraph(name='mult')

    # Adding a HierarchalTensorGraph
    edges = [('input', mult), (mult, 'output')]
    m.add_edges_from(edges)
    assert m({'scalar': 2, 'x': 3}) == {'product': 6}

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

def test_remove_node():
    m = AddMultExp()
    m.remove_node('add_mult')
    assert 'add_mult' not in m.nodes 
    for i in m.edges:
        assert 'add_mult' not in i

def test_remove_edge():
    m = AddMultExp()
    m.remove_edge('input','add_mult')
    assert ('input','add_mult') not in m.edges 

def test_simple_coupled_model():
    m = AddMult()
    X = {'x_1': 0.2, 'x_2': 0.3, 'scalar': 1.5}
    res = m(X)
    assert res['add_mult_res'] == 0.75

    n = AddMultExp()
    res = n(X)
    assert pytest.approx(res['add_mult_exp_res']) == exp(0.75)

def test_same_name_different_model():
    def fa(s):
        print('fa')

    def fb(s):
        print('fb')

    with pytest.raises(ImproperTensorGraphError):
        m1 = Node(node=fa,inputs={},outputs={},name='a')
        m2 = Node(node=fb,inputs={},outputs={},name='a')
        m3 = HierarchalTensorGraph(name='b')
        m3.add_edge(m1, m2)


def test_duplicate_models():
    def fa(s):
        print('fa')

    def fb(s):
        print('fb')

    with pytest.raises(ImproperTensorGraphError):
        m1 = Node(node=fa,inputs={},outputs={},name='a')
        m2 = Node(node=fb,inputs={},outputs={},name='b')
        m3 = HierarchalTensorGraph(name='c')
        m3.add_edge(m1, m2)
        m4 = HierarchalTensorGraph(name='d')
        m4.add_edge(m1, m3)


def test_add_edge_to_basenode():
    with pytest.raises(ImproperTensorGraphError):
        m = Node(node=lambda x:x,inputs={},outputs={},name='a')
        m.add_edge('input', 'output')

def test_output_node_exist():
    n = Node(node=lambda x:x,inputs={},outputs={},name='a')
    m = HierarchalTensorGraph(name="m")
    m.add_edge('input',n)
    with pytest.raises(ImproperTensorGraphError):
        m({'x': 1})


def test_input_is_not_target():
    n = Node(node=lambda x:x,inputs={},outputs={},name='a')
    m = HierarchalTensorGraph(name="m")
    with pytest.raises(ImproperTensorGraphError):
        m.add_edge(n, 'input')


def test_output_is_not_source():
    n = Node(node=lambda x:x,inputs={},outputs={},name='a')
    m = HierarchalTensorGraph(name="m")
    with pytest.raises(ImproperTensorGraphError):
        m.add_edge('output', n)


def test_no_key_found():
    n = Node(
        node=lambda X :{'out' : X['a']},
        inputs={'a' : None},
        outputs={'out' : None},
        name='a'
    )

    m = HierarchalTensorGraph(name="m")
    m.add_edge('input', n)
    m.add_edge(n,'output')
    with pytest.raises(ImproperTensorGraphError):
        m({'b': 1})

def test_multiple_keys_found():

    m1 = Node(
        node=lambda X: {'f1': X['x']},
        name="m1",
        inputs={'x': None},
        outputs={'f1': None}
    )

    m2 = Node(
        node=lambda X: {'f1': X['x']},
        name="m2",
        inputs={'x': None},
        outputs={'f1': None}
    )

    n = HierarchalTensorGraph(name="n")
    n.add_edge('input', m1)
    n.add_edge('input', m2)

    with pytest.raises(ImproperTensorGraphError):
        n.add_edge(m1, m2)

def test_hanging_source_nodes():
    m1 = Node(
        node=lambda X: {'x' : X['x']},
        name="m1",
        inputs={'x': None},
        outputs={'x': None}
    )


    model = HierarchalTensorGraph(name='test')
    model.add_edge('input','output')
    model.add_node(m1)
    with pytest.raises(ImproperTensorGraphError):
         model({})

def test_hanging_sinks_nodes():
    m1 = Node(
        node=lambda X: {'x' : X['x']},
        name="m1",
        inputs={'x': None},
        outputs={'x': None}
    )

    m2 = Node(
        node=lambda X: {'x' : X['x']},
        name="m2",
        inputs={'x': None},
        outputs={'x': None}
    )

    model = HierarchalTensorGraph(name='test')
    model.add_edge('input', m1)
    model.add_edge('input', m2)
    model.add_edge(m2, 'output')
    with pytest.raises(ImproperTensorGraphError):
        model({'x': 2})


def test_HTG_as_basenode():
    m = Node(
        node=lambda X: {'x' : X['x']},
        name="m",
        inputs={'x': None},
        outputs={'x': None}
    )

    with pytest.raises(ImproperTensorGraphError):
        Node(
            node=m,
            name="m1",
            inputs={'x': None},
            outputs={'x': None}
        )

    m = HierarchalTensorGraph(name='m')
    with pytest.raises(ImproperTensorGraphError):
        Node(
            node=m,
            name="m1",
            inputs={'x': None},
            outputs={'x': None}
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
        inps = (TensorSpec(node_1.inputs[k]).spec_dict == TensorSpec(
            node_2.inputs[k]).spec_dict) and inps

    if not inps:
        print(node_1.inputs.keys(), node_2.inputs.keys())
        print(TensorSpec(node_1.inputs['x']).spec_dict,
              node_2.inputs['x'].spec_dict)

    outs = node_1.outputs.keys() == node_2.outputs.keys()

    for k in node_1.outputs.keys():
        outs = (TensorSpec(node_1.outputs[k]).spec_dict == TensorSpec(
            node_2.outputs[k]).spec_dict) and outs

    if not outs:
        print(TensorSpec(node_1.outputs['y']).spec_dict,
              node_2.outputs['y'].spec_dict)

    edge = node_1.edges == node_2.edges

    if not edge:
        print(node_1.edges, node_2.edges)

    inpmap = node_1._inputs_map == node_2._inputs_map

    if not inpmap:
        print(node_1._inputs_map, node_2._inputs_map)

    outmap = node_1._outputs_map == node_2._outputs_map

    if not outmap:
        print(node_1._outputs_map, node_2._outputs_map)

    return inps and outs and edge and inpmap and outmap

@pytest.mark.skip
def test_node_json():
    node = Node(**{
        'node' : lambda X: {'add' : X['x'] + 10},
        'name' : 'add',
        'inputs' : {'x': (1,)},
        'outputs' : {'add': (1,)}
    })

    jnode = Node.from_json(node.to_json())
    assert(node({'x' : 2}) == jnode({'x' : 2}))

@pytest.mark.skip
def test_json_basic():

    node = Adder()
    htg = HierarchalTensorGraph(name='htg_add')
    htg.add_edge('input', node)
    htg.add_edge(node, 'output')
    fhtg = HierarchalTensorGraph.from_json(htg.to_json())
    assert (fhtg( {'x_1': 1, 'x_2': 2}) == htg({'x_1': 1, 'x_2': 2}))

#def test_json_model():
#    keras_node = AddKerasModel()
#
#    htg_1 = HierarchalTensorGraph(name='test_json_model')
#    htg_1.add_edge('input', keras_node)
#    htg_1.add_edge(keras_node, 'output')
#
#    htg_1.inputs = {'x': None}
#    htg_1.outputs = {'y': None}
#
#    json_data = htg_1.to_json()
#
#    htg_2 = HierarchalTensorGraph.from_json(json_data)
#    assert (not htg_2 == None)
#
#    assert (htg_1.edges == htg_2.edges)
#    assert (htg_1.nodes['test'].node.get_config() ==
#            htg_2.nodes['test'].node.get_config())
#

@pytest.mark.skip
def test_json_AddSquare():
    htg = AddSquare()
    fhtg = HierarchalTensorGraph.from_json(htg.to_json())
    X = {'x_1' : 1, 'x_2' : 2}
    assert(fhtg(X) == htg(X))

@pytest.mark.skip
def test_json_LogAddSquare():
    htg = LogAddSquare()
    fhtg = LogAddSquare.from_json(htg.to_json())
    X = {'x_1': 1, 'x_2': 2}
    print(fhtg(X))
    #assert (fhtg(X) == htg(X))

@pytest.mark.skip
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


@pytest.mark.skip
def test_logjson_custom():

    htg = HierarchalTensorGraph(
        name='log',
        node=LogCustom()
    )

    json_data = htg.to_json()

    HierarchalTensorGraph.from_json(json_data)


def test_recurrent_single():
    htg = SingleRecurrent()
    result = htg({'x_1': np.array([[1, 2, 3]])})
    assert ('s_1' in result)
    assert(np.any(np.array(result['s_1']) == np.array([[1],[3],[6]])))


def test_recurrent_single_order():
    htg = IdentityPlusSingleRecurrent()
    result = htg({'x_1': np.array([[1, 2, 3]])})
    assert(np.any(np.array(result['s_1']) == np.array([[1],[3],[6]])))

def test_recurrent_dual():
    htg = DualRecurrent()
    result = htg({'x_1': np.array([[1, 2, 3]]), 'x_2': np.array([[1, 2, 3]])})
    answer = {'s_1': np.array([6]), 's_2': np.array([10])}
    for k in result: assert(answer[k] == result[k])

def test_recurrent_triple():
    htg = TripleRecurrent()
    result = htg({
        'x_1': np.array([[1, 2, 3]]),
        'x_2': np.array([[1, 4, 5]]),
        'x_3': np.array([[3, 2, 6]])
    })
    answer = {'s_1': np.array([26]), 's_2': np.array([28]), 's_3': np.array([29])}
    for k in result: assert(answer[k] == result[k])

def test_recurrent_renaming():
    sr = SingleRecurrent()
    htg = HierarchalTensorGraph(name='rename')
    htg.add_edge('input',sr)
    htg.add_edge(sr,'output')
    result = htg({'x_1': np.array([[1, 2, 3]])})
    assert(np.any(np.array(result['s_1']) == np.array([[1],[3],[6]])))

def test_key_labels():
    def f(X):
        return X 

    node = Node(**{
        'name' : 'labels',
        'node' : lambda x : f(x),
        'inputs' : {'ifs__x' : 1, 
                    'ifs__y' : 1 , 
                    'era5__x' : 1, 
                    'y' : 1
                    },
        'outputs' : {'ifs__x' : 1, 
                    'ifs__y' : 1,
                    'era5__x' : 1, 
                    'y' : 1
                    },
    })

    htg = HierarchalTensorGraph('labels')
    htg.add_edge('input',node)
    htg.add_edge(node,'output',labels='ifs')
    res = htg({'ifs__x' : 1, 
               'ifs__y' : 2, 
               'era5__x' : 3,
               'y' : 4}
               )
    # print(res)
    assert res == {'ifs__x' : 1, 'ifs__y' : 2}
    # return
    htg.remove_edge(node,'output')
    htg.add_edge(node,'output',labels='era5')
    res = htg({'ifs__x' : 1, 
               'ifs__y' : 2, 
               'era5__x' : 3,
               'y' : 4}
               )
    assert res == {'era5__x' : 3}
    
    htg.remove_edge(node,'output')
    htg.add_edge(node,'output',labels=['ifs','era5'])
    res = htg({'ifs__x' : 1, 
               'ifs__y' : 2, 
               'era5__x' : 3,
               'y' : 4}
               )
    assert res == {'ifs__x' : 1, 'ifs__y' : 2, 'era5__x' : 3}
    
    htg.remove_edge(node,'output')
    htg.add_edge(node,'output',features=['y'],labels=['ifs'])
    res = htg({'ifs__x' : 1, 
               'ifs__y' : 2, 
               'era5__x' : 3,
               'y' : 4}
               )
    assert res == {'ifs__x' : 1, 'ifs__y' : 2, 'y' : 4}

    # Test grouped inputs 
    def f(X):
        return {**X['ifs'],**X['era5']}

    node = Node(**{
        'name' : 'groups',
        'node' : lambda x : f(x),
        'inputs' : {'ifs__x' : 1, 
                    'ifs__y' : 1 , 
                    'era5__z' : 1, 
                    't' : 1
                    },
        'outputs' : {'x':1, 'y':2, 'z':3},
        **{'grouped_inputs' : True}
    })

    res = node({'ifs__x' : 1, 
               'ifs__y' : 2, 
               'era5__z' : 3,
               't' : 4}
               )
    htg = HierarchalTensorGraph('groups')
    htg.add_edge('input',node)
    htg.add_edge(node,'output')
    res = htg({'ifs__x' : 1, 
               'ifs__y' : 2, 
               'era5__z' : 3,
               't' : 4}
               )
    assert res == {'x': 1, 'y': 2, 'z': 3}

def test_registry():

    class Child(HierarchalTensorGraph):
        registry_name = 'Children' 
        pass

    class Baby(Child):
        pass

    assert(Child.registry_name == 'Children')
    assert(Baby.registry_name == 'Baby')
