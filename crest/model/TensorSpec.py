import json
import traceback
import tensorflow as tf

# is parallel to keras.TensorSpec


class TensorSpec(tf.TensorSpec):
    def __init__(self, *specs):

        self.spec_dict = {}

        if (len(specs) == 1 and isinstance(specs[0], list)):
            self.spec_dict['shape'] = specs[0]
            self.spec_dict['dtype'] = 'float32'
            self.spec_dict['name'] = None

        elif (len(specs) == 1 and isinstance(specs[0], tuple)):
            self.spec_dict['shape'] = specs[0]
            self.spec_dict['dtype'] = 'float32'
            self.spec_dict['name'] = None

        # specs is a dictionary
        elif (len(specs) == 1 and isinstance(specs[0], dict)):
            self.spec_dict['shape'] = specs[0]['shape']
            self.spec_dict['dtype'] = specs[0]['dtype']
            self.spec_dict['name'] = specs[0]['name']

        # specs is a tuple
        elif (len(specs) >= 1):
            self.spec_dict['shape'] = specs[0]

            # optional parameter
            if (len(specs) >= 2):
                self.spec_dict['dtype'] = specs[1]
            else:
                # default to float32
                self.spec_dict['dtype'] = 'float32'

            # optional parameter
            if (len(specs) >= 3):
                self.spec_dict['name'] = specs[2]
            else:
                self.spec_dict['name'] = None

        else:
            raise Exception(
                'TensorSpec must be initialized with a tuple or dict')

        print(self.spec_dict)

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
    def keras(self):
        import tensorflow as tf

        return tf.TensorSpec(
            shape=self.shape,
            dtype=tf.dtypes.as_dtype(self.dtype),
            name=self.name)

    def __eq__(self, other):
        same_keys = (self.specs.keys() == other.specs.keys())

        if (same_keys):
            if (self.specs['shape'] == tuple(other.specs['shape'])):
                if (self.specs['dtype'] == other.specs['dtype']):
                    if (self.specs['name'] == other.specs['name']):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

        else:
            return False

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
