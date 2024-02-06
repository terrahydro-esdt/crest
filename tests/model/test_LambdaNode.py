from crest.model.LambdaNode import LambdaNode

def setup_lambdanode():
    node=lambda X: {'mult': X['scalar'] * X['x']}
    name='mult'
    inputs={'scalar': None, 'x': None}
    outputs={'mult': None}
    
    return LambdaNode(node, name=name, inputs=inputs, outputs=outputs)

def test_init():
    node = setup_lambdanode()
 
    assert node.name == 'mult'
    assert node.inputs == {'scalar': None, 'x': None}
    assert node.outputs == {'mult': None}
    assert node({'scalar': 2, 'x': 3}) == {'mult': 6}

def test_tofromjson():
    node = setup_lambdanode()

    json_obj = node.to_json()
    new_node = LambdaNode.from_json(json_obj)

    assert new_node.name == 'mult'
    assert new_node.inputs == {'scalar': None, 'x': None}
    assert new_node.outputs == {'mult': None}
    assert new_node({'scalar': 2, 'x': 3}) == {'mult': 6}

def test_save():
    import os

    node = setup_lambdanode()
    node.save()

    assert os.path.exists('lambda.json')

    os.remove('lambda.json')

def test_load():
    import os

    node = setup_lambdanode()
    node.save()

    assert os.path.exists('lambda.json')

    new_node = LambdaNode.load('lambda.json')

    assert new_node.name == 'mult'
    assert new_node.inputs == {'scalar': None, 'x': None}
    assert new_node.outputs == {'mult': None}
    assert new_node({'scalar': 2, 'x': 3}) == {'mult': 6}

    os.remove('lambda.json')
    