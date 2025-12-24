import logging
from contextlib import nullcontext
import numpy as np
import warnings
import itertools
import os
import json
from keras import Input
from keras import Model as KerasModel
from keras.models import load_model
from crest.utils import Metrics
from crest.data.loading import Dataset, StructuredDataset
from crest.data import Batcher
from .BaseModel import BaseModel, ImproperModelError
from .TensorGraph import TensorGraph
from crest.model.TensorSpec import TensorSpec
from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
from crest.utils.save_node_class import write_pkl, read_pkl, gen_filename

logger = logging.getLogger(__name__)


class Model(BaseModel):
    """Builds tensor models using HierarchalTensorGraph

    Parameters
    ----------
    graph: TensorGraph
        The Hierarchal TensorGraph

    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __init__(self, graph: TensorGraph, **kwargs):
        self.graph = graph
        self.name = graph.name

        self.graph.build()

        # Allow for TensorSpec to be converted to Keras Input
        self.inputs = {}
        for k, v in self.graph.inputs.items():
            if (not v is None):
                if (isinstance(v, TensorSpec)):
                    v = v.tf
                # Compatibility with future TF versions
                try:
                    self.inputs[k] = Input(type_spec=v, name=k)
                except:
                    self.inputs[k] = Input(shape=v.shape[1:], dtype=v.dtype, name=k)
            else:
                self.inputs[k] = None

        self.outputs = self.graph(self.inputs)
        self.metric = Metrics()
        self.model = KerasModel(inputs=self.inputs,outputs=self.outputs)
         
        # Explicitly set model output_names for correct logging labels
        if isinstance(self.outputs, dict):
            self.model.output_names = sorted(self.outputs)

    def compile(self, show_summary: bool = False, **kwargs):
        """
        Builds and compiles the Keras model. Inputs and outputs
        are set as specified in `self.graph`.

        Parameters
        ----------
        show_summary : bool
            Whether to show the model summary after compilation.
        **kwargs
            args passed as keras.Model.compile(kwargs)

        """
        logger.debug(f'Compiling Keras Model')

        # Check kwargs for metrics locally defined
        # if 'metrics' in kwargs:
        #     metrics = self.metric.get_callbacks(kwargs['metrics'])
        #     kwargs['metrics'] = metrics

        self.model.compile(**kwargs)

        # Verify there are trainable weights in this model
        no_trainable = len(self.model.trainable_weights) <= 0
        if show_summary or no_trainable:
            self.model.summary(line_length=200)
        if no_trainable:
            self.logger.warning('\nNo trainable parameters in model\n')

    def fit(self, data, **kwargs):
        """
        Fit the Model.

        Parameters
        ----------

        dataset : The training data containing both inputs and targets. Dataset
        can be either a crest.batcher or anything keras.model.fit takes.

        kwargs : Keyword args for the fitter. Currently, can be any keyword args
        accepted by keras.fit().

        """
        logger.debug(f'Training Keras Model')

        if isinstance(data,Batcher):
            with data as batcher, kwargs.get('validation_data', nullcontext()):
                if 'validation_data' in kwargs:
                    kwargs['validation_data'] = iter(kwargs['validation_data'])
                self.model.fit(iter(batcher), **kwargs)
        else:
            self.model.fit(data, **kwargs)

    def predict(self, dataset, coords: str | list =[], **kwargs) -> dict:
        """
        Make predictions with the model.

        Parameters
        ----------

        dataset : The inputs to make predictions on. Dataset
        can be either a crest.Batcher or anyting Keras.fit takes.
        If using a crest.Batcher, setting steps == 'all' will
        exhaust the batcher (repeat must be false).

        coords: The names/keys of additional features to include in the output.
        The keys must be contained in batcher along with the input. Used
        only with a crest.Batcher.

        kwargs : Keyword args for prediciton. Currently, can be any keyword args
        accepted by Keras.predict().

        """
        logger.debug(f'Predicting with Keras Model')

        # Check if exhaust is possible
        steps = None

        # Make sure coords is a list
        if isinstance(coords, str):
            coords = [coords]

        if(coords and not isinstance(dataset,Batcher)):
            message = 'The arg coords can only be used when'\
                      ' dataset is a crest.Batcher'
            raise ValueError(message)
        
        # Make prediction using Keras.predict()
        if isinstance(dataset,Batcher):
            pred = []
            lbls = []
            steps = None

            # Get steps
            if 'steps' in kwargs:
                steps = kwargs.pop('steps')
            
            with dataset as data:
                
                if not steps:
                    steps = data
                    if dataset.repeat:
                        raise ImproperModelError(
                                'When using a crest.Batcher you must. '\
                                'specify steps if repeat == False'
                                )
                else:
                    steps = itertools.islice(data,steps)


                for batch in steps:
                    pred_batch = self.predict_on_batch(batch,coords,**kwargs)
                    if pred:
                        for key in pred_batch.keys():
                            pred[key] = np.concatenate([pred[key], pred_batch[key]])
                    if not pred:
                        pred = pred_batch
            return pred

        return self.model.predict(dataset,**kwargs)

    def predict_on_batch(self, batch: dict, coords=[], **kwargs) -> dict:
        """
        Make predictions with the model.

        Parameters
        ----------

        dataset : The inputs to make predictions on. Dataset is a dictionary.

        coords: The names/keys of additional features to include in the output.
        The keys must be contained in dataset along with the input.

        kwargs : Keyword args for prediciton. Currently, can be any keyword args
        accepted by Keras.predict().

        """
        # Make sure coords is a list
        if isinstance(coords, str):
            coords = [coords]

        # Make prediction using Keras.predict()
        pred = []
        lbls = []

        # Pop out auxillary outputs
        if coords:
            lbls = {i: batch.pop(i) for i in coords}
        pred_batch = self.model.predict_on_batch(batch)

        # Add additional coords
        if coords:
            for k, v in lbls.items():
                pred_batch[k] = v

        if pred:
            for key in pred_batch.keys():
                pred[key] = np.concatenate([pred[key], pred_batch[key]])
        if not pred:
            pred = pred_batch

        return pred

    def evaluate(self, x=None, y=None, **kwargs):
        """
        Evaluate the performance of the fitter

        Parameters
        ----------
      
        Evaulate wraps keras.evaluate(). See, Keras docs.

        """

        if isinstance(x,Batcher):
            x = iter(x)
            y = None

        return self.model.evaluate(x,y, **kwargs)

    def save_weights(self):
        """
        Save the weights of the model

        """
        logger.debug(f'Save weights')

        os.makedirs('crest_cache', exist_ok=True)
        self.model.save_weights('crest_cache/htg.weights.h5')

    def load_weights(self, path: str):
        """
        Load the weights of the model

        """
        logger.debug(f'Load weights')

        self.model.load_weights(os.path.join(path, 'htg.weights.h5'))

    def save_model(self, model_path):
        """
        Save model by registering the keras model

        """
        logger.debug(f'Save model as custom_model')

        if hasattr(self.model, 'save') and callable(self.model.save):
            from tensorflow.keras.utils import get_custom_objects

            logger.debug('Saving model with tensorflow keras.')

            get_custom_objects()[model_path] = self.model
            self.model.save(model_path)
        else:
            logger.debug("Saving model using pickle.")

            if (os.path.isdir(model_path)):
                pickle_name = gen_filename(self.name, '.pkl')
                pickle_name = os.path.join(model_path, pickle_name)
            
            write_pkl(self.model, pickle_name)

            logger.debug(f"Saved metadata to the same root path: {Path(model_path).parent}")
            

    @staticmethod
    def load_model(model_type: str = 'keras', path: str = 'htg_model', custom_objects: dict = None):
        """
        Load model registered as custom object

        """
        root_dir = Path(path).parent

        logger.debug(f'Load custom model from {root_dir}')

        try:
            if (model_type == 'keras'):
                logger.debug('Loading model using Tensorflow Keras library.')

                if (custom_objects is None):
                    loaded = tf.keras.models.load_model(path)
                else:
                    loaded = tf.keras.models.load_model(path, custom_objects)
                
                return loaded
            else:
                logger.debug('Loading model using pickle.')

                model_obj = read_pkl(root_dir)
                
                logger.debug(f'Model object loaded.')
                return model_obj
        except Exception as e:
            logger.error(f'Could not load model from CREST Model: {e}')
            raise ImproperModelError('Cannot recreate previous CREST Model')

    def save(self, dir='crest_cache', save_metrics=False):
        """
        Converts the model to a json string

        """
        logger.debug(f'Save Keras Model')

        os.makedirs(dir, exist_ok=True)

        # convert graph to json
        graph_json = self.graph.to_json()

        with open(os.path.join(dir, 'htg.graph.json'), 'w') as f:
            json.dump(graph_json, f)

        # convert model to json
        self.save_model(dir)

        if (save_metrics):
            # covert metrics to json
            logger.debug(self.metric.customs)

            metric_json = self.metric.to_json()

            logger.debug(metric_json)

            with open(os.path.join(dir, 'htg.metric.json'), 'w') as f:
                json.dump(metric_json, f)

    @staticmethod
    def load(path: str, model_type: str, load_metrics=False):
        """
        Converts the model from a json string.

        Parameters
        ----------
        json_str : str
            The json string to convert the model from.

        """
        # convert graph from json
        with open(os.path.join(path, 'htg.graph.json'), 'r') as f:
            graph_json = json.load(f)
            graph = HierarchalTensorGraph.from_json(graph_json)

        model = Model(graph)

        if (load_metrics):
            logger.debug(load_metrics)
            metrics = None

            # convert metrics from json
            with open(os.path.join(path, 'htg.metric.json'), 'r') as f:
                metric_json = json.load(f)
                metrics = Metrics.from_json(metric_json)

            custom_metrics = {v: metrics.get_handler(
                v) for v in metrics.customs}
            
            model.model = Model.load_model(model_type, path, custom_metrics)
        else:
            model.model = Model.load_model(model_type, path)

        # convert model from json
        model.model = load_model(os.path.join(path, 'htg.model.h5'), custom_objects=custom_metrics)

        return model
