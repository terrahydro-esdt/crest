from crest.model.KerasNode import KerasNode, KerasNodeType
from crest.model.TensorSpec import TensorSpec
import json

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


def setup_seqmodel():
    model = Sequential()
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def setup_kerasmodel():
    layers = [keras.layers.Dense(32, activation="relu"), keras.layers.Dropout(0.5), keras.layers.Dense(10, activation="softmax")]
    inputs = tf.keras.Input(shape=(32, ),name='x')

    outputs = inputs
    for layer in layers:
        outputs = layer(outputs)

    model = tf.keras.Model(inputs={'x': inputs}, outputs={'y': outputs})
 
    return model

def setup_keraslayers():
    return Dense(32)

def test_init():
    model = setup_kerasmodel()

    node = KerasNode(model, name='test', inputs={"x": tf.TensorSpec(
        [None, 32], tf.float32,name='x')}, outputs={"y": tf.TensorSpec([None, 10], tf.float32,name='y')})

    inp = TensorSpec((None, 32), 'float32', None)
    out = TensorSpec((None, 10), 'float32', None)

    assert node.inputs["x"].shape == inp.shape
    assert node.inputs["x"].dtype == inp.dtype

    assert node.outputs["y"].shape == out.shape
    assert node.outputs["y"].dtype == out.dtype
    assert node.keras_obj == model

def test_inputs_outputs():
    model = setup_kerasmodel()

    node = KerasNode(model, name='test', inputs={"x": tf.TensorSpec(
        [None, 32], tf.float32)}, outputs={"y": tf.TensorSpec([None, 10], tf.float32)})

    assert node.inputs["x"] == TensorSpec(tf.TensorSpec((None, 32), tf.float32))
    assert node.outputs["y"] == TensorSpec(tf.TensorSpec((None, 10), tf.float32))
    assert node.keras_obj == model

def test_layer():
    layer = setup_keraslayers()
    inputs = {"x": tf.TensorSpec([None, 784], tf.float32)}
    outputs = {"y": tf.TensorSpec([None, 32], tf.float32)}
    
    node = KerasNode(layer, name='test', inputs=inputs, outputs=outputs)

    assert node.inputs['x'].spec_dict == {'shape': (None, 784), 'dtype': 'float32', 'name': None}
    assert node.outputs['y'].spec_dict == {'shape': (None, 32), 'dtype': 'float32', 'name': None}
    assert isinstance(node.keras_obj, keras.Model)
    assert isinstance(node.keras_obj.layers[1], keras.layers.Dense)

def test_seqmodel():
    model = setup_seqmodel()

    inputs = {"x": tf.TensorSpec([None, 784], tf.float32)}
    outputs = {"y": tf.TensorSpec([None, 10], tf.float32)}
    
    node = KerasNode(model, name='test', inputs=inputs, outputs=outputs)

    assert node.inputs['x'] == TensorSpec((None, 784), 'float32')
    assert node.outputs['y'] == TensorSpec((None, 10), 'float32')
    assert isinstance(node.keras_obj, keras.Model)

def test_tojson():
    model = setup_kerasmodel()

    node = KerasNode(model, name='test')

    json_str = node.to_json()
    json_obj = json.loads(json_str)

    assert json_obj['type'] == KerasNodeType.MODEL.value

    loaded_node = KerasNode.from_json(json_str)

    assert TensorSpec.dict_to_json(loaded_node.inputs) == TensorSpec.dict_to_json(node.inputs)
    assert TensorSpec.dict_to_json(loaded_node.outputs) == TensorSpec.dict_to_json(node.outputs)

    assert isinstance(loaded_node.keras_obj, keras.Model)
    assert isinstance(loaded_node.keras_obj.layers[1], keras.layers.Dense)
    assert isinstance(loaded_node.keras_obj.layers[2], keras.layers.Dropout)
    assert isinstance(loaded_node.keras_obj.layers[3], keras.layers.Dense)
