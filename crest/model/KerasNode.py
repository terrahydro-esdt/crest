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
    """
    Enum for the different types of keras objects that can be used in KerasNode
    """
    MODEL = 0
    LAYER = 1
    SEQUENTIAL = 2


class KerasNode(HierarchalTensorGraph):
    """
    KerasNode is a wrapper around a keras model or layer that allows for
    individuals to define a custom basenode of type HierarchalTensorGraph. 
    This allows for easy saving and loading of basenodes with keras models 
    and layers within the CREST framework. 

    Args:
        keras_obj: The keras model or layer to be wrapped
        name: The name of the node
        inputs: A dictionary of the input tensor specs
        outputs: A dictionary of the output tensor specs
    """
    def __init__(self, keras_obj, name: None | str = None, inputs: dict = {}, outputs: dict = {}):
        """
        Initializes the KerasNode with the given keras object, name, inputs, and outputs.
        """

        # Check the type of the keras object
        self.type = self._check_type(keras_obj)

        # If the keras object is a model
        if (self.type == KerasNodeType.MODEL):
            self.keras_obj = keras_obj

            # Input and output must be dictionaries
            # TODO: Double check that this is the correct way to handle inputs and outputs
            if (not isinstance(self.keras_obj.input, dict)):
                raise ImproperModelError(
                    'Keras model must take type dictionary for inputs')

            if (not isinstance(self.keras_obj.output, dict)):
                raise ImproperModelError(
                    'Keras model must take type dictionary for outputs')

            # Set the inputs and outputs to the dictionary of CREST tensor specs
            self.inputs = {k: TensorSpec(v)
                           for k, v in self.keras_obj.input.items()
                           }
            self.outputs = {k: TensorSpec(v)
                            for k, v in self.keras_obj.output.items()
                            }

        # If the keras object is a layer
        elif (self.type == KerasNodeType.LAYER):

            # Input and output must be defined and only have one key-value pair.
            # This is the assumption of this library. 
            if (not inputs or not outputs):
                raise Exception(
                    'KerasNode must be initialized with inputs and outputs for keras layers')

            if (len(inputs.keys()) > 1 or len(outputs.keys()) > 1):
                raise Exception(
                    'KerasNode must be initialized with only one input or output for keras layers')

            # Set the inputs and outputs to the dictionary of CREST tensor specs
            self.inputs = {k: TensorSpec(v) for k, v in inputs.items()}
            self.outputs = {k: TensorSpec(v) for k, v in outputs.items()}

            # Create a keras model with the given layer
            key_x = list(self.inputs.keys())[0]
            key_y = list(self.outputs.keys())[0]
            x = tf.keras.Input(type_spec=self.inputs[key_x].tf)
            y = keras_obj(x)

            self.keras_obj = keras.Model(inputs={key_x: x}, outputs={key_y: y})

        # If the keras object is a sequential model
        elif (self.type == KerasNodeType.SEQUENTIAL):

            # Input and output must be defined and only have one key-value pair.
            if (not inputs or not outputs):
                raise Exception(
                    'KerasNode must be initialized with inputs and outputs for keras layers')

            if (len(inputs.keys()) > 1 or len(outputs.keys()) > 1):
                raise Exception(
                    'KerasNode must be initialized with only one input or output for keras layers')

            # Set the inputs and outputs to the dictionary of CREST tensor specs
            self.inputs = {k: TensorSpec(v) for k, v in inputs.items()}
            self.outputs = {k: TensorSpec(v) for k, v in outputs.items()}

            # Create a keras model with the given layer
            key_x = list(self.inputs.keys())[0]
            key_y = list(self.outputs.keys())[0]
            x = tf.keras.Input(type_spec=self.inputs[key_x].tf)
            y = keras_obj(x)

            self.keras_obj = keras.Model(inputs={key_x: x}, outputs={key_y: y})

        # If the keras object is not a model, layer, or sequential model then raise an exception
        else:
            raise Exception(
                'KerasNode must be initialized with a keras model or layer')

        super().__init__(self.keras_obj, name, self.inputs, self.outputs)

    def _check_type(self, obj):
        """
        Checks the type of the given keras object and returns the corresponding KerasNodeType

        Args:
            obj: The keras object to check the type of

        Returns:
                KerasNodeType: The type of the given keras object
        """
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
        """
        Create JSON string of the KerasNode

        Returns:
            str: JSON string of the KerasNode
        """
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
        """
        Create a KerasNode from a JSON string

        Args:
            keras_json: The JSON string to create the KerasNode from
        
        Returns:
            KerasNode: The KerasNode created from the JSON string
        """
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
        """
        Save the KerasNode to a file

        Args:
            path: The path to save the KerasNode to        
        """
        self.keras_obj.save(path)
        
    @staticmethod
    def load(path='model.keras'):
        """
        Load a KerasNode from a file

        Args:
            path: The path to load the KerasNode from
        """
        model = keras.load_model(path)
        return KerasNode(model)
