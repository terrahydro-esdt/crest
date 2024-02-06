from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
import tensorflow.keras as keras
import traceback
from enum import Enum
from pydoc import locate
import tensorflow as tf
import tensorflow.keras as keras
from .TensorSpec import TensorSpec
from .BaseModel import ImproperModelError
import json


class KerasNodeType(Enum):
    MODEL = 0
    LAYER = 1
    SEQUENTIAL = 2


class KerasNode(HierarchalTensorGraph):
    def __init__(self, keras_obj, name: None | str = None, inputs: dict = {}, outputs: dict = {}):
        self.type = self._check_type(keras_obj)

        if (self.type == KerasNodeType.MODEL):
            self.keras_obj = keras_obj

            # TODO: Double check that this is the correct way to handle inputs and outputs
            if (not isinstance(self.keras_obj.input, dict)):
                raise ImproperModelError(
                    'Keras model must take type dictionary for inputs')

            if (not isinstance(self.keras_obj.output, dict)):
                raise ImproperModelError(
                    'Keras model must take type dictionary for outputs')

            self.inputs = {k: TensorSpec(v)
                           for k, v in self.keras_obj.input.items()
                           }
            self.outputs = {k: TensorSpec(v)
                            for k, v in self.keras_obj.output.items()
                            }

        elif (self.type == KerasNodeType.LAYER):
            if (not inputs or not outputs):
                raise Exception(
                    'KerasNode must be initialized with inputs and outputs for keras layers')

            if (len(inputs.keys()) > 1 or len(outputs.keys()) > 1):
                raise Exception(
                    'KerasNode must be initialized with only one input or output for keras layers')

            # make sure the shape of the output layer is the same
            self.inputs = {k: TensorSpec(v) for k, v in inputs.items()}
            self.outputs = {k: TensorSpec(v) for k, v in outputs.items()}

            key_x = list(self.inputs.keys())[0]
            key_y = list(self.outputs.keys())[0]
            x = tf.keras.Input(type_spec=self.inputs[key_x].tf)
            y = keras_obj(x)

            self.keras_obj = keras.Model(inputs={key_x: x}, outputs={key_y: y})
        elif (self.type == KerasNodeType.SEQUENTIAL):
            if (not inputs or not outputs):
                raise Exception(
                    'KerasNode must be initialized with inputs and outputs for keras layers')

            if (len(inputs.keys()) > 1 or len(outputs.keys()) > 1):
                raise Exception(
                    'KerasNode must be initialized with only one input or output for keras layers')

            self.inputs = {k: TensorSpec(v) for k, v in inputs.items()}
            self.outputs = {k: TensorSpec(v) for k, v in outputs.items()}

            key_x = list(self.inputs.keys())[0]
            key_y = list(self.outputs.keys())[0]
            x = tf.keras.Input(type_spec=self.inputs[key_x].tf)
            y = keras_obj(x)

            self.keras_obj = keras.Model(inputs={key_x: x}, outputs={key_y: y})
        else:
            raise Exception(
                'KerasNode must be initialized with a keras model or layer')

        super().__init__(self.keras_obj, name, self.inputs, self.outputs)

    def _check_type(self, obj):
        if (isinstance(obj, keras.models.Sequential)):
            return KerasNodeType.SEQUENTIAL
        elif (isinstance(obj, keras.Model)):
            return KerasNodeType.MODEL
        elif (isinstance(obj, keras.layers.Layer)):
            return KerasNodeType.LAYER
        else:
            raise ImproperModelError(
                'Improper model type: %s' % str(type(obj)))

    def to_json(self):
        keras_json = self.__dict__.copy()

        keras_json.pop('graph')
        keras_json.pop('keras_obj')
        keras_json.pop('output')

        keras_json['node'] = self.keras_obj.to_json()
        keras_json['node_class'] = self.__class__.__name__
        keras_json['node_module'] = self.__module__
        keras_json['inputs'] = TensorSpec.dict_to_json(self.inputs)
        keras_json['outputs'] = TensorSpec.dict_to_json(self.outputs)

        keras_json['type'] = keras_json['type'].value

        return json.dumps(keras_json)

    @staticmethod
    def from_json(keras_json):
        keras_dict = keras_json
        if (isinstance(keras_json, str)):
            keras_dict = json.loads(keras_json)

        model = keras.models.model_from_json(keras_dict['node'])

        inputs = TensorSpec.json_to_dict(keras_dict['inputs'])
        outputs = TensorSpec.json_to_dict(keras_dict['outputs'])

        kn = KerasNode(
            model, keras_dict['name'], inputs, outputs)

        for k, v in keras_dict.items():
            if k not in ['node', 'inputs', 'outputs']:
                setattr(kn, k, v)

        return kn

    def save(self, path='model.keras'):
        if (self.type == KerasNodeType.MODEL):
            self.keras_obj.save(path)
            return True
        else:
            raise Exception(
                'Could not save keras object: not a model (%s)' % str(type(self.keras_obj)))

    @staticmethod
    def load(path='model.keras'):
        model = keras.load_model(path)
        return KerasNode(model)
