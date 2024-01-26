from contextlib import nullcontext

import tensorflow as tf
import numpy as np
import warnings

from crest.utils import Metrics
from crest.data.loading import Dataset, StructuredDataset
from crest.data import Batcher
from .BaseModel import BaseModel, ImproperModelError
from .TensorGraph import TensorGraph
from crest.model.TensorSpec import TensorSpec


class Model(BaseModel):
    """Builds tensor models using HierarchalTensorGraph

    Parameters
    ----------
    graph: TensorGraph
        The Hierarchal TensorGraph

    """

    def __init__(self, graph: TensorGraph, **kwargs):
        self.graph = graph
        self.model = None
        self.name  = graph.name

        # Allow for TensorSpec to be converted to Keras Input
        self.inputs = {}
        for k,v in self.graph.inputs.items():
            if (not v is None):
                if (isinstance(v, TensorSpec)):
                    v = v.keras

                self.inputs[k] = tf.keras.Input(type_spec=v, name=k)
            else:
                self.inputs[k] = None

        self.outputs = self.graph(self.inputs)
        self.metric = Metrics()


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
        if isinstance(dataset,Batcher):
            return dataset

        if isinstance(dataset,dict):
            ds = StructuredDataset(*list(dataset.values()),labels=list(dataset.keys()))
            return Batcher(ds,**kwargs)

        if isinstance(dataset,(Dataset,StructuredDataset)):
            return Batcher(dataset,**kwargs)

        message = f'Cannot make a Batcher from {type(dataset)}: it must be either a Dataset, StructuredDataset, '
        message += 'Batcher, or dictionary.'
        raise ImproperModelError(message)


    def build(self, _internal=False, **kwargs):
        """ Builds Keras model """
        if not _internal:
            warnings.warn('Use Model.compile instead of Model.build')
            return self.compile(**kwargs)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        # Explicitly set model output_names for correct logging labels
        if isinstance(self.outputs, dict):
            self.model.output_names = sorted(self.outputs)


    def compile(self, show_summary : bool = False, **kwargs):
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
        # Check kwargs for metrics locally defined
        if ('metrics' in kwargs):
            metrics = self.metric.get_callbacks(kwargs['metrics'])
            kwargs['metrics'] = metrics

        self.build(_internal=True)
        self.model.compile(**kwargs)

        # Verify there are trainable weights in this model
        no_trainable = len(self.model.trainable_weights) <= 0
        if show_summary or no_trainable:
            self.model.summary(line_length=200)
        if no_trainable: print('\nWARNING: No trainable parameters in model\n')


    def fit(self, dataset : Dataset | Batcher | StructuredDataset, **kwargs):
        """
        Fit the Model.

        Parameters
        ----------

        dataset : The training data containing both inputs and targets. Dataset
        can be either a Dataset, StructuredDataset, Batcher, or dict.

        kwargs : Keyword args for the fitter. Currently, can be any keyword args
        accepted by Keras.fit().

        """

        # Add default options for kwargs if necessary
        defaults = {'steps_per_epoch': 1,
                    'epochs'         : 1,
                    'batch_size'     : 1,
                    'shuffle'        : True
                   }

        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Training Batcher
        train_kwargs = {
            'batch_size' : kwargs['batch_size'],
            'features'   : [list(self.inputs),list(self.outputs)],
            'repeat'     : True,
            'shuffle'    : kwargs['shuffle']
        }

        training_batcher = self._make_batcher(dataset,**train_kwargs)

        # Validation Batcher
        if 'validation_data' in kwargs:

            if 'validation_steps' not in kwargs:
                kwargs['validation_steps'] = 1

            valid_kwargs = {
                'batch_size' : kwargs ['batch_size'],
                'features'   : [list(self.inputs),list(self.outputs)],
                'shuffle'    : False
            }

            kwargs['validation_data'] = self._make_batcher(kwargs['validation_data'],**valid_kwargs)

        with training_batcher as data, kwargs.get('validation_data',nullcontext()):
            self.model.fit(data,**kwargs)


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

        # Set default options for kwargs
        defaults = {
            'batch_size'  : 1,
            'steps'       : 1
        }

        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Make sure coords is a list
        if isinstance(coords,str):
            coords = [coords]

        # Prediction Batcher
        batch_kwargs = {
            'batch_size' : kwargs ['batch_size'],
            'features'   : coords + list(self.inputs),
            'shuffle'    : False
        }

        batcher = self._make_batcher(dataset,**batch_kwargs)

        # Make prediction using Keras.predict()
        pred = []
        lbls = []
        with batcher as data:
            steps = kwargs.pop('steps')
            for i in range(steps):
                batch = next(data)
                # Pop out auxillary outputs
                if coords: lbls = {i : batch.pop(i) for i in coords}
                pred_batch = self.model.predict(batch, **kwargs)
                # Add additional coords
                if coords:
                    for k,v in lbls.items(): pred_batch[k] = v

                if pred:
                    for key in pred_batch.keys():
                        pred[key] = np.concatenate([pred[key],pred_batch[key]])
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
        if isinstance(coords,str):
            coords = [coords]

        # Make prediction using Keras.predict()
        pred = []
        lbls = []

        # Pop out auxillary outputs
        if coords: lbls = {i : batch.pop(i) for i in coords}
        pred_batch = self.model.predict_on_batch(batch)

        # Add additional coords
        if coords:
            for k,v in lbls.items(): pred_batch[k] = v

        if pred:
            for key in pred_batch.keys():
                pred[key] = np.concatenate([pred[key],pred_batch[key]])
        if not pred:
            pred = pred_batch

        return pred


    def predict_exhaust(self, dataset : Dataset | StructuredDataset | Batcher | dict, coords=[], **kwargs) -> dict:
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

        # Set default options for kwargs
        defaults = {
            'batch_size'  : 1,
            'steps'       : 1,
            'exhaust'      : False
        }

        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Make sure coords is a list
        if isinstance(coords,str):
            coords = [coords]

        # Prediction Batcher
        batch_kwargs = {
            'batch_size' : kwargs ['batch_size'],
            'features'   : coords + list(self.inputs),
            'shuffle'    : False
        }

        batcher = self._make_batcher(dataset,**batch_kwargs)

        if (kwargs.get('exhaust') and batcher.repeat):
            raise ImproperModelError('Cannot exhaust batcher when batcher is set to repeat.')

        # Make prediction using Keras.predict()
        with batcher as data:
            steps = kwargs.pop('steps')
            exhaust = kwargs.pop('exhaust')

            # collect all batches if exhaust else collect number of batches specified by steps
            batches = list(data) if(exhaust) else [batch for _, batch in zip(range(steps), data)]

            # concatenate all batches based on keys
            batch = {}
            for k in batches[0].keys():
                batch[k] = np.concatenate([b[k] for b in batches])

            # for each prediction of each batch
            # Pop out auxillary outputs
            lbls = None
            if coords:
                lbls = {i : batch.pop(i) for i in coords}

            pred_batches = self.model.predict(batch, **kwargs)

            if coords:
                for k,v in lbls.items():
                    pred_batches[k] = v

        return pred_batches


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
        if isinstance(coords,str):
            coords = [coords]

        # Make prediction using Keras.predict()
        pred = []
        lbls = []

        # Pop out auxillary outputs
        if coords: lbls = {i : batch.pop(i) for i in coords}
        pred_batch = self.model.predict_on_batch(batch)
        # Add additional coords
        if coords:
            for k,v in lbls.items(): pred_batch[k] = v

        if pred:
            for key in pred_batch.keys():
                pred[key] = np.concatenate([pred[key],pred_batch[key]])
        if not pred:
            pred = pred_batch

        return pred


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

        # Default options since we use generators
        defaults = {
            'batch_size'  : 1,
            'steps'       : 1
        }

        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        # Evaluation Batcher
        batch_kwargs = {
            'batch_size' : kwargs['batch_size'],
            'features'   : [list(self.inputs),list(self.outputs)],
            'shuffle'    : False
         }

        batcher = self._make_batcher(dataset,**batch_kwargs)
        with batcher as data:
            return self.model.evaluate(data, **kwargs)
