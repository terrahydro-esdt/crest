import logging
from contextlib import nullcontext

import tensorflow as tf
import numpy as np
import warnings
import os
import json
import inspect
from pathlib import Path

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
        self.model = None
        self.name = graph.name

        logger.info(f'Initializing CREST Model')

        if not graph.inputs or not graph.outputs:
            raise ImproperModelError(
                'Inputs and outputs must be specified for HierarchalTensorGraph')

        # Allow for TensorSpec to be converted to Keras Input
        self.inputs = {}
        for k, v in self.graph.inputs.items():
            if (not v is None):
                if (isinstance(v, TensorSpec)):
                    v = v.tf
                # Compatibility with future TF versions
                try:
                    self.inputs[k] = tf.keras.Input(type_spec=v, name=k)
                except: 
                    self.inputs[k] = tf.keras.Input(shape=v.shape[1:], dtype=v.dtype, name=k)
            else:
                self.inputs[k] = None

        self.outputs = self.graph(self.inputs)
        self.metric = Metrics()

        logger.info(f'Completed Initializing CREST Model')

    def _make_batcher(self, dataset, **kwargs) -> Batcher:
        """
        Makes a Batcher

        Parameters
        ----------

        dataset : The data to make into a Batcher. Dataset
        can be either a Dataset, StructuredDataset, Batcher, or dict.
        If Batcher is passed, it is returned.

        kwargs: kwargs to pass to the Batcher

        """
        logger.info(f'Called _make_batcher, making batcher from dataset')

        if isinstance(dataset, Batcher):
            return dataset

        if isinstance(dataset, dict):
            ds = StructuredDataset(
                *list(dataset.values()), labels=list(dataset.keys()))
            return Batcher(ds, **kwargs)

        if isinstance(dataset, (Dataset, StructuredDataset)):
            return Batcher(dataset, **kwargs)

        message = f'Cannot make a Batcher from {type(dataset)}: it must be either a Dataset, StructuredDataset, '
        message += 'Batcher, or dictionary.'
        raise ImproperModelError(message)

    def build(self, _internal=False, **kwargs):
        """ Builds Keras model """
        logger.info(f'Building Keras Model')

        if not _internal:
            warnings.warn('Use Model.compile instead of Model.build')
            return self.compile(**kwargs)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

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
        logger.info(f'Compiling Keras Model')

        # Check kwargs for metrics locally defined
        # if 'metrics' in kwargs:
        #     metrics = self.metric.get_callbacks(kwargs['metrics'])
        #     kwargs['metrics'] = metrics

        self.build(_internal=True)
        self.model.compile(**kwargs)

        # Verify there are trainable weights in this model
        no_trainable = len(self.model.trainable_weights) <= 0
        if show_summary or no_trainable:
            self.model.summary(line_length=200)
        if no_trainable:
            self.logger.warning('\nNo trainable parameters in model\n')

    def fit(self, dataset: Dataset | Batcher | StructuredDataset, **kwargs):
        """
        Fit the Model.

        Parameters
        ----------

        dataset : The training data containing both inputs and targets. Dataset
        can be either a Dataset, StructuredDataset, Batcher, or dict.

        kwargs : Keyword args for the fitter. Currently, can be any keyword args
        accepted by Keras.fit().

        """
        logger.info(f'Training Keras Model')

        # Add default options for kwargs if necessary
        defaults = {'steps_per_epoch': 1,
                    'epochs': 1,
                    'batch_size': 1,
                    'shuffle': True
                    }

        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Training Batcher
        train_kwargs = {
            'batch_size': kwargs['batch_size'],
            'features': [list(self.inputs), list(self.outputs)],
            'repeat': True,
            'shuffle': kwargs['shuffle'],
            'workers': 0
        }

        training_batcher = self._make_batcher(dataset, **train_kwargs)

        # Validation Batcher
        if 'validation_data' in kwargs:

            if 'validation_steps' not in kwargs:
                kwargs['validation_steps'] = 1

            valid_kwargs = {
                'batch_size': kwargs['batch_size'],
                'features': [list(self.inputs), list(self.outputs)],
                'shuffle': False,
                'workers': 0
            }

            kwargs['validation_data'] = self._make_batcher(
                kwargs['validation_data'], **valid_kwargs)

            kwargs['validation_data'] = self._make_batcher(
                kwargs['validation_data'], **valid_kwargs)

        with training_batcher as data, kwargs.get('validation_data', nullcontext()):
            self.model.fit(data, **kwargs)

    def predict(self, dataset: Dataset | StructuredDataset | Batcher | dict, coords=[], **kwargs) -> dict:
        """
        Make predictions with the model.

        Parameters
        ----------

        dataset : The inputs to make predictions on. Dataset
        can be either a Dataset, StructuredDataset, Batcher, or dict.

        coords: The names/keys of additional features to include in the output.
        The keys must be contained in dataset along with the input.

        kwargs : Keyword args for prediciton. Currently, can be any keyword args
        accepted by Keras.predict().

        """
        logger.info(f'Predicting with Keras Model')

        # Set default options for kwargs
        defaults = {
            'batch_size': 1,
            'steps': 1
        }

        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Make sure coords is a list
        if isinstance(coords, str):
            coords = [coords]

        # Prediction Batcher
        batch_kwargs = {
            'batch_size': kwargs['batch_size'],
            'features': coords + list(self.inputs),
            'shuffle': False
        }

        batcher = self._make_batcher(dataset, **batch_kwargs)

        # Make prediction using Keras.predict()
        pred = []
        lbls = []
        with batcher as data:
            steps = kwargs.pop('steps')
            for i in range(steps):
                batch = next(data)
                # Pop out auxillary outputs
                if coords:
                    lbls = {i: batch.pop(i) for i in coords}
                pred_batch = self.model.predict(batch, **kwargs)
                # Add additional coords
                if coords:
                    for k, v in lbls.items():
                        pred_batch[k] = v

                if pred:
                    for key in pred_batch.keys():
                        pred[key] = np.concatenate(
                            [pred[key], pred_batch[key]])
                if not pred:
                    pred = pred_batch
        return pred

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

    def predict_exhaust(self, dataset: Dataset | StructuredDataset | Batcher | dict, coords=[], **kwargs) -> dict:
        """
        Make predictions with the model.

        Parameters
        ----------

        dataset : The inputs to make predictions on. Dataset
        can be either a Dataset, StructuredDataset, Batcher, or dict.

        coords: The names/keys of additional features to include in the output.
        The keys must be contained in dataset along with the input.

        kwargs : Keyword args for prediciton. Currently, can be any keyword args
        accepted by Keras.predict().

        """
        logger.info(f'Predicting exhaustive')

        # Set default options for kwargs
        defaults = {
            'batch_size': 1,
            'steps': 1,
            'exhaust': False
        }

        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Make sure coords is a list
        if isinstance(coords, str):
            coords = [coords]

        # Prediction Batcher
        batch_kwargs = {
            'batch_size': kwargs['batch_size'],
            'features': coords + list(self.inputs),
            'shuffle': False
        }

        batcher = self._make_batcher(dataset, **batch_kwargs)

        if kwargs.get('exhaust') and batcher.repeat:
            raise ImproperModelError(
                'Cannot exhaust batcher when batcher is set to repeat.')

        # Make prediction using Keras.predict()
        with batcher as data:
            steps = kwargs.pop('steps')
            exhaust = kwargs.pop('exhaust')

            # collect all batches if exhaust else collect number of batches specified by steps
            batches = list(data) if exhaust else [
                batch for _, batch in zip(range(steps), data)]

            # concatenate all batches based on keys
            batch = {}
            for k in batches[0].keys():
                batch[k] = np.concatenate([b[k] for b in batches])

            # for each prediction of each batch
            # Pop out auxillary outputs
            lbls = None
            if coords:
                lbls = {i: batch.pop(i) for i in coords}

            pred_batches = self.model.predict(batch, **kwargs)

            if coords:
                for k, v in lbls.items():
                    pred_batches[k] = v

        return pred_batches

    def evaluate(self, dataset: Dataset | Batcher | StructuredDataset | dict, **kwargs):
        """
        Evaluate the performance of the fitter

        Parameters
        ----------

        dataset : The dataset containing inputs and outputs. Dataset
        can be either a Dataset, StructuredDataset, Batcher, or dict.

        kwargs : Keyword args for evaluation. Currently, can be any keyword args
        accepted by Keras.evaluate().

        """

        logger.info(f'Evaluating Keras Model')

        # Default options since we use generators
        defaults = {
            'batch_size': 1,
            'steps': 1
        }

        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Evaluation Batcher
        batch_kwargs = {
            'batch_size': kwargs['batch_size'],
            'features': [list(self.inputs), list(self.outputs)],
            'shuffle': False
        }

        batcher = self._make_batcher(dataset, **batch_kwargs)
        with batcher as data:
            return self.model.evaluate(data, **kwargs)

    def save_weights(self):
        """
        Save the weights of the model

        """
        logger.info(f'Save weights')

        os.makedirs('crest_cache', exist_ok=True)
        self.model.save_weights('crest_cache/htg.weights.h5')

    def load_weights(self, path: str):
        """
        Load the weights of the model

        """
        logger.info(f'Load weights')

        self.model.load_weights(os.path.join(path, 'htg.weights.h5'))

    def save_model(self, model_path):
        """
        Save model by registering the keras model

        """
        logger.info(f'Save model as custom_model')

        if hasattr(self.model, 'save') and callable(self.model.save):
            from tensorflow.keras.utils import get_custom_objects

            logger.info('Saving model with tensorflow keras.')

            get_custom_objects()[model_path] = self.model
            self.model.save(model_path)
        else:
            logger.info("Saving model using pickle.")

            if (os.path.isdir(model_path)):
                pickle_name = gen_filename(self.name, '.pkl')
                pickle_name = os.path.join(model_path, pickle_name)
            
            write_pkl(self.model, pickle_name)

            logger.info(f"Saved metadata to the same root path: {Path(model_path).parent}")
            

    @staticmethod
    def load_model(model_type: str = 'keras', path: str = 'htg_model', custom_objects: dict = None):
        """
        Load model registered as custom object

        """
        root_dir = Path(path).parent

        logger.info(f'Load custom model from {root_dir}')

        try:
            if (model_type == 'keras'):
                logger.info('Loading model using Tensorflow Keras library.')

                if (custom_objects is None):
                    loaded = tf.keras.models.load_model(path)
                else:
                    loaded = tf.keras.models.load_model(path, custom_objects)
                
                return loaded
            else:
                logger.info('Loading model using pickle.')

                model_obj = read_pkl(root_dir)
                
                logger.info(f'Model object loaded.')
                return model_obj
        except Exception as e:
            logger.error(f'Could not load model from CREST Model: {e}')
            raise ImproperModelError('Cannot recreate previous CREST Model')

    def save(self, dir='crest_cache', save_metrics=False):
        """
        Converts the model to a json string

        """
        logger.info(f'Save Keras Model')

        os.makedirs(dir, exist_ok=True)

        # convert graph to json
        graph_json = self.graph.to_json()

        with open(os.path.join(dir, 'htg.graph.json'), 'w') as f:
            json.dump(graph_json, f)

        # convert model to json
        self.save_model(dir)

        if (save_metrics):
            # covert metrics to json
            print(self.metric.customs)

            metric_json = self.metric.to_json()

            print(metric_json)

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
            print(load_metrics)
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

        return model
