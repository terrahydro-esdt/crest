from .BaseModel import BaseModel,ImproperModelError
from .TensorGraph import TensorGraph
from ..data.loading import Dataset,StructuredDataset
from ..data.Batcher import Batcher
from tensorflow.keras import Input
from tensorflow.keras import Model as KerasModel
from numpy import concatenate
from contextlib import nullcontext


class Model(BaseModel):
    """Builds tensor models using HierarchalTensorGraph
    
    Parameters
    ----------
    
    TensorGraph : The HierarchalTensorGraph
    
    """

    def __init__(self,
                 TensorGraph : TensorGraph,
                 **kwargs):
        self.TensorGraph = TensorGraph
        self.model = None
        self.name = TensorGraph.name
        self.inputs = {k : Input(type_spec=v)  if not v is None else 
                       None for k,v in self.TensorGraph.inputs.items()}
        self.outputs = self.TensorGraph(self.inputs)
        
    def _make_batcher(self,dataset,**kwargs) -> Batcher:
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
            
        message = 'Cannot make a Batcher form dataset. It must be either a Dataset, StructuredDataset, '
        message += 'Batcher, or dictionary.'
        raise ImproperModelError(message)
        
    def build(self,**kwargs):
        '''
        builds and compiles the Keras model. Inputs and outputs
        are set as specified in TensorGraph.

        Parameters
        ----------

        kwargs : args passed as keras.Model.compile(kwargs)

        '''

        self.model = KerasModel(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(**kwargs)

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

    def predict(self, dataset : Dataset | Batcher | dict, coords=[], **kwargs) -> dict:
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
                        pred[key] = concatenate([pred[key],pred_batch[key]])
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
                pred[key] = concatenate([pred[key],pred_batch[key]])
        if not pred: 
            pred = pred_batch
            
        return pred

    def evaluate(self, dataset : Dataset | Batcher | StructuredDataset | dict, **kwargs) -> dict:
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
