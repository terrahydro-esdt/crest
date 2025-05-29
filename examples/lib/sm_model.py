import tensorflow as tf
from crest.utils.save_node_class import *

class sm_model(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(sm_model, self).__init__(**kwargs)
        self._temporal = tf.keras.layers.LSTM(units=256, name='temporal')
        self._head = tf.keras.layers.Dense(1, activation='relu')
        self._features = {
            'ERA5'       : ['cape', 'cp', 'csfr','cvh', 'cvl', 'e', 'es',
                            'fal','lai_hv', 'lai_lv', 'msdwlwrf', 'msdwswrf',
                            'pev', 'sd','skt','smlt','sp', 'sro','slhf',
                            'sshf','ssro','stl1', 'stl2', 'stl3', 'stl4',
                            't2m','tp','u10', 'v10', 'z'],
            'Irrigation' : ['Actual', 'Equipped', 'GroundWater', 
                            'Non-Conventional', 'SurfaceWater'],
            'Soil'       : ['Clay_frac', 'Sand_frac', 'Silt_frac'],
            'SMAP'       : ['soil_moisture']
        }

    def call(self, X):
        era5 = tf.stack([X[k] for k in self._features['ERA5']], axis=-1)
        era5 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[2, 3]))(era5)
        x = self._temporal(era5)

        soil = tf.stack([X[k] for k in self._features['Soil']], axis=-1)
        soil = tf.squeeze(tf.cast(soil, dtype=tf.float32), axis=[1, 2])

        irrigation = tf.stack([X[k] for k in self._features['Irrigation']], axis=-1)
        irrigation = tf.squeeze(tf.cast(irrigation, dtype=tf.float32), axis=[1, 2])

        x = tf.concat([x, soil, irrigation], axis=-1)
        output = {'soil_moisture': self._head(x)}
        return output

    def get_config(self):
        config = super(sm_model, self).get_config()
        return config

    def to_json(self, filepath='sm_model.pkl'):
        config_dict = tf.keras.utils.serialize_keras_object(self)

        filepath = os.path.join('outputs', filepath)            

        write_pkl(self, filepath)
        config_dict['filepath'] = filepath
        return config_dict

    @staticmethod
    def from_json(json_input):
        filepath = json_input['filepath']
        obj = read_pkl(filepath, pathname=__file__)
        return obj