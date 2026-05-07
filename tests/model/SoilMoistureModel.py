import dill
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM
from crest.nodes.tensorflow import Node
from crest.data.transform import Transform
from crest.model import IOSpec


class SoilMoistureModel(Node):
    # Specify registry name (defaults is __class__.name)
    registry_name = 'TestSoilMoistureModel'

    # Define inputs
    inputs_spec = IOSpec()
    inputs_spec['ERA5'] = {
        'keys' : ['cape', 'cp', 'csfr','cvh', 'cvl', 'e', 'es',
                  'fal','lai_hv', 'lai_lv', 'msdwlwrf', 'msdwswrf',
                  'pev', 'sd','skt','smlt','sp', 'sro','slhf',
                  'sshf','ssro','stl1', 'stl2', 'stl3', 'stl4',
                  't2m','tp','u10', 'v10', 'z'],
        'coord_shapes' : {'datetime' : None, 'latitude' : 1, 'longitude' : 1}
        }

    inputs_spec['Irrigation'] = {
        'keys' : ['Actual', 'Equipped', 'GroundWater','Non-Conventional', 'SurfaceWater'],
        'coord_shapes' : {'latitude' : 1, 'longitude' : 1}
        }
    
    inputs_spec['Soil'] = {
        'keys' : ['Clay_frac', 'Sand_frac', 'Silt_frac'],
        'coord_shapes' : {'latitude' : 1, 'longitude' : 1}
        }

    # Define output spec 
    outputs_spec = IOSpec({'SMAP' : {
        'keys': 'soil_moisture', 
        'coord_shapes' : {'datetime' : 1, 'latitude' : 1, 'longitude' : 1}
        }})
            
    def __init__(self,stats):
        self.stats = stats
        transforms = Transform(self.stats,by_feature={'*' : 'logexp'})
        super().__init__('SM',transforms.standardize)
        self._temporal = LSTM(units=256, name='temporal')
        self._head = Dense(1, activation='relu')

    def call(self,X,training=True):
        era5 = tf.stack(list(X['ERA5'].values()), axis=-1)
        era5 = tf.squeeze(era5, axis=[2,3])
        x = self._temporal(era5)
        soil = tf.stack(list(X['Soil'].values()), axis=-1)
        soil = tf.squeeze(tf.cast(soil, dtype=float), axis=[1,2])
        irrigation = tf.stack(list(X['Irrigation'].values()), axis=-1)
        irrigation = tf.squeeze(tf.cast(irrigation, dtype=float), axis=[1,2])
        x = tf.concat([x, soil, irrigation], axis=-1)
        return {'SMAP>>soil_moisture' : self._head(x)}