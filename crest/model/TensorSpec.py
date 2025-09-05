import json
import traceback
import tensorflow as tf
from tensorflow import keras

# is parallel to keras.TensorSpec


class TensorSpec(object):
    def __init__(self, *specs):

        self.spec_dict = {}
        print(f'{specs=}')

        if len(specs) == 1 and isinstance(specs[0], TensorSpec):
            self.spec_dict['shape'] = self._modify_shape(specs[0].shape)
            self.spec_dict['dtype'] = specs[0].dtype
            self.spec_dict['name'] = specs[0].name

        elif len(specs) == 1 and hasattr(specs[0], 'type_spec') and isinstance(specs[0].type_spec, tf.TensorSpec):
            self.spec_dict['shape'] = self._modify_shape(
                specs[0].type_spec.shape)
            self.spec_dict['dtype'] = specs[0].type_spec.dtype.name
            self.spec_dict['name'] = specs[0].type_spec.name

        elif len(specs) == 1 and isinstance(specs[0], list):
            self.spec_dict['shape'] = self._modify_shape(specs[0])
            self.spec_dict['dtype'] = 'float32'
            self.spec_dict['name'] = None

        elif len(specs) == 1 and isinstance(specs[0], tuple):
            self.spec_dict['shape'] = self._modify_shape(specs[0])
            self.spec_dict['dtype'] = 'float32'
            self.spec_dict['name'] = None

        elif len(specs) == 1 and isinstance(specs[0], dict):
            self.spec_dict['shape'] = self._modify_shape(specs[0]['shape'])
            self.spec_dict['dtype'] = specs[0]['dtype']
            self.spec_dict['name'] = specs[0]['name']

        elif len(specs) == 1 and isinstance(specs[0], tf.TensorSpec):
            self.spec_dict['shape'] = self._modify_shape(specs[0].shape)
            self.spec_dict['dtype'] = specs[0].dtype.name
            self.spec_dict['name'] = specs[0].name

        # specs is a tuple
        elif len(specs) >= 1:
            self.spec_dict['shape'] = self._modify_shape(specs[0])

            # optional parameter
            if len(specs) >= 2:
                self.spec_dict['dtype'] = specs[1]
            else:
                # default to float32
                self.spec_dict['dtype'] = 'float32'

            # optional parameter
            if len(specs) >= 3:
                self.spec_dict['name'] = specs[2]
            else:
                self.spec_dict['name'] = None

        else:
            raise Exception(
                'TensorSpec must be initialized with a tuple or dict')

    def _modify_shape(self, shape):
        if isinstance(shape, tf.TensorShape):
            return tuple(shape.as_list())

        if isinstance(shape,keras.KerasTensor):
            return tuple(shape.shape)

        return shape

    @property
    def specs(self):
        return self.spec_dict

    @property
    def shape(self):
        return tuple(self.spec_dict['shape'])

    @property
    def dtype(self):
        return self.spec_dict['dtype']

    @property
    def name(self):
        return self.spec_dict['name']

    @property
    def tf(self):
        return tf.TensorSpec(
            shape=self.shape,
            dtype=tf.dtypes.as_dtype(self.dtype),
            name=self.name)

    @property
    def keras(self):
        return keras.Input(
            shape=self.shape,
            dtype=self.dtype,
            name=self.name)

    @staticmethod
    def dict_to_json(htg_dict: dict):

        str_json = {}
        for k, v in htg_dict.items():
            if v:
                str_json[k] = TensorSpec(v).spec_dict
            else:
                str_json[k] = None

        return json.dumps(str_json)

    @staticmethod
    def json_to_dict(str_json: dict):

        str_dict = json.loads(str_json)
        htg_dict = {}

        for k, v in str_dict.items():
            if v:
                htg_dict[k] = TensorSpec(v)
            else:
                htg_dict[k] = None

        return htg_dict

    def __eq__(self, other):
        same_keys = (self.specs.keys() == other.specs.keys())

        if same_keys:
            if self.specs['shape'] == tuple(other.specs['shape']):
                if self.specs['dtype'] == other.specs['dtype']:
                    return True
                else:
                    return False
            else:
                return False

        else:
            return False

    def __str__(self):
        return str(self.specs)

    def save(self, path):
        try:
            with open(path, 'w') as f:
                json.dump(self.specs, f)

            return True
        except:
            print(traceback.format_exc())
            return False

    @staticmethod
    def load(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return TensorSpec(data)
        except:
            print(traceback.format_exc())
            return None
