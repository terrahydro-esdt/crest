from crest.base import BaseNode
from crest.data.transform.Transform import Transform
import xarray as xr
import tensorflow as tf


class SoilMoistureModel(BaseNode):
    """ A model for estimating the SMAP soil moisture value using Three
        data sources: 1- Time series of environmental factors like
        precipitation and wind speed (stacked), 2- Static attributes
        including: Soil types, Topography, Irrigation, 3- Landcover,
        and vegetation indices as well as the MODIS surface reflectance
        in red, blue, MIR and NIR.

        Parameters
        ----------
        stats : xr.DataArray
            A DataArray of the summary statistics of the variables of the 
            dataset
    """

    inputs = {
        'ERA5.zarr' : {
            'cape': {
               'datetime': None, 
               'latitude': 1, 
               'longitude': 1
            }, # Convective Available Potential Energy
        },

        'StaticAttributes.zarr/Soil' : {
            'Clay_frac' : {'latitude': 1, 'longitude': 1},
            'Sand_frac' : {'latitude': 1, 'longitude': 1},
            'Silt_frac' : {'latitude': 1, 'longitude': 1},
        },

        'StaticAttributes.zarr/Irrigation': {
            'Actual'           : {'latitude': 3, 'longitude': 3},
            'Equipped'         : {'latitude': 3, 'longitude': 3},
            'GroundWater'      : {'latitude': 3, 'longitude': 3},
            'Non-Conventional' : {'latitude': 3, 'longitude': 3},
            'SurfaceWater'     : {'latitude': 3, 'longitude': 3},
        },        

    }

    outputs = {
        'ERA5.zarr' : {
            'swvl1' : {'datetime': 1, 'latitude': 1, 'longitude': 1},
        },
    }



    def __init__(self, stats: xr.DataArray):
        # These features should not be transformed, since they're classes / flags
        must_not_transform = ['cvh']        
        self.stats = stats.where(~stats.features.isin(must_not_transform), drop=True)
        super().__init__(*Transform(self.stats).normalize,
                         loss='mse',
                         transform_loss=True)
        layer_kwargs = {
            'lstm_size'          : 256,
            'dropout'            : 0.2,
        }

        self._spatial_irrigation = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, depth_multiplier=1),
            tf.keras.layers.PReLU(),
        ], name='spatial_irrigation')

        self._temporal = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x, [2, 3])),
            tf.keras.layers.LSTM(units=layer_kwargs['lstm_size']),
            tf.keras.layers.Dropout(layer_kwargs['dropout']),
        ], name='temporal')

        self._model  = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.PReLU(),

            tf.keras.layers.Dense(1),
        ], name='SM-Model')


    def call(self, X):

        # capture ERA5 variables
        era5 = X.pop('ERA5.zarr')
        era5 = tf.stack(list(era5.values()), axis=-1)

        # capture soil variables
        soil = tf.stack([v
            for source, features in X.items()
            for v in features.values()
            if 'Soil' in source
        ], axis=-1)
        soil = tf.expand_dims(soil, axis=1)
        soil = tf.repeat(soil, int(tf.shape(era5)[1]), axis=1)

        # capture irrigation variables
        irrigation = tf.stack([v
            for source, features in X.items()
            for v in features.values()
            if 'Irrigation' in source], axis=-1)
        irrigation = self._spatial_irrigation(irrigation)
        irrigation = tf.expand_dims(irrigation, axis=1)
        irrigation = tf.repeat(irrigation, int(tf.shape(era5)[1]), axis=1)


        # concatenate all input vars and run the LSTM
        sm = self._model(self._temporal(tf.concat([era5,
                                                   soil,
                                                   irrigation], axis=-1)))

        return {'swvl1': sm}
 