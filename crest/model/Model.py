import zipfile
import base64
import tempfile
import logging
import numpy as np
import warnings
import itertools
import os
import json

from contextlib import nullcontext
from pathlib import Path
from keras import Input
from keras import Model as KerasModel
from keras.models import load_model

from .BaseModel import BaseModel, ImproperModelError
from .TensorGraph import TensorGraph
from .TensorSpec import TensorSpec
from .HierarchalTensorGraph import HierarchalTensorGraph
from crest.utils import Metrics
from crest.data.loading import Dataset, StructuredDataset
from crest.data import Batcher


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

    def __init__(self, graph: HierarchalTensorGraph):
        self.graph = graph
        self.name = graph.name

        self.graph.build()

        # Allow for TensorSpec to be converted to Keras Input
        self.inputs = {}
        for k, v in self.graph.inputs.items():
            if not v is None:
                if isinstance(v, TensorSpec):
                    v = v.tf
                # Compatibility with future TF versions
                try:
                    self.inputs[k] = Input(type_spec=v, name=k)
                except:
                    self.inputs[k] = Input(shape=v.shape[1:], dtype=v.dtype, name=k)
            else:
                self.inputs[k] = None
        
        self.outputs = self.graph(self.inputs)
        self.metrics = Metrics()
        self._model = KerasModel(inputs=self.inputs,outputs=self.outputs)

        # Explicitly set model output_names for correct logging labels
        if isinstance(self.outputs, dict):
            self._model.output_names = sorted(self.outputs)

    @property
    def model(self):
        return self._model

    def __call__(self,X):
        """ 
        
        Call the model 
        
        Parameters
        ----------

        X : dictionary of tensors
        
        """
        return self.model(X)

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

        if isinstance(x, Batcher):
            with x as batcher:
                return self.model.evaluate(iter(batcher), None, **kwargs)

        return self.model.evaluate(x, y, **kwargs)
    
    def get_weights(self):
        """ Returns a numpy array of the weights of the mdoel"""
        return self.model.get_weights()
    
    def set_weights(self,weights):
        """ Returns a numpy array of the weights of the mdoel"""
        return self.model.set_weights(weights)

    def save_weights(self,filepath, overwrite=True, max_shard_size=None):
        """ Save the weights of the model """

        logger.debug(f'Saving weights')
        self.model.save_weights(filepath,overwrite,max_shard_size)

    def load_weights(self,filepath,skip_mismatch=False, **kwargs):
        """ Load the weights of the model """

        logger.debug(f'Load weights')
        self.model.load_weights(filepath,skip_mismatch,**kwargs)

    def save(self,filepath,overwrite=True):
        """ 
        Saves the crest model into a zipfile. This
        saves the:

        - HierarchalTensorGraph
        - trainable weights
        - compile args
        - Any metrics

        It reconstructs the htg and underlying model. Reloads the weights,
        compile args, and metrics. Note, it does not save the optimizer
        state.

        Parameters
        ----------

         filepath : str or pahtlib.Path object where the model is saved.
         
        """

        logger.debug(f'Saving model')

        if isinstance(filepath,str):
            filepath = Path(filepath)

        # Create temporary directory and zip
        with tempfile.TemporaryDirectory() as temp_dir:

            # Convert to base64 and save htg. Also, handle HTG node encoding correctly
            raw_encode = self.graph.encode()
            raw_nodes = raw_encode.pop("nodes", None)  # None -> key absent
            encode = {
                k: base64.b64encode(v).decode("utf-8") for k, v in raw_encode.items()
            }
            if raw_nodes is not None:
                encode["nodes"] = [
                    {
                        k: (
                            base64.b64encode(v).decode("utf-8")
                            if isinstance(v, bytes)
                            else v
                        )
                        for k, v in node.items()
                    }
                    for node in raw_nodes
                ]
            encode["crest_registry_name"] = self.graph.registry_name

            with open("graph.json",'w') as f:
                json.dump(encode,f)
           
            # Save weights 
            self.save_weights('model.weights.h5')

            # Save metrices 
            with open("metrics.json",'w') as f:
                json.dump(self.metrics.to_json(),f)
           
            # Save keras compile args
            with open("compile_args.json",'w') as f:
                json.dump(self.model.get_compile_config(),f)
            
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write('graph.json')
                zipf.write('model.weights.h5')
                zipf.write('metrics.json')
                zipf.write('compile_args.json')

    @classmethod 
    def load(cls,filepath):
        """ 
        Loads a saved crest model.

        Parameters
        ----------

         filepath : str or pahtlib.Path of the saved model.
         
        """

        logger.debug(f'Saving model')

        if isinstance(filepath,str):
            filepath = Path(filepath)

        # Create temporary directory and zip
        with tempfile.TemporaryDirectory() as temp_dir:

            # Unzip file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
           
            # Load config and create Model
            with open("graph.json",'r') as f:
                graph = json.load(f)

            registry_name = graph.pop("crest_registry_name")
            # Here we need to handle HTG node encoding correctly
            raw_nodes = graph.pop("nodes", None)  # None -> key absent
            decode = {k: base64.b64decode(v) for k, v in graph.items()}
            if raw_nodes is not None:
                decode["nodes"] = [
                    {
                        k: (
                            base64.b64decode(v)
                            if isinstance(v, str) and k != "crest_registry_name"
                            else v
                        )
                        for k, v in node.items()
                    }
                    for node in raw_nodes
                ]
            htg_obj = HierarchalTensorGraph.registry[registry_name]
            htg = htg_obj.decode(decode)
            obj = cls(htg)

            # Load weights
            obj.load_weights('model.weights.h5')

            # Load metrices 
            with open("metrics.json",'r') as f:
                metrics = json.load(f)

            obj.metrics = Metrics.from_json(metrics)
           
           # Load keras compile args
            with open("compile_args.json",'r') as f:
                compile_args = json.load(f)

            obj.compile(**compile_args)

            return obj

