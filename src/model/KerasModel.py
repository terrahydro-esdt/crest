from .Model import Model,ImproperModelError
from .TensorGraph import TensorGraph
from ..data.loading import Dataset,StructuredDataset
from ..data.Batcher import Batcher
from tensorflow.keras import Input
from tensorflow.keras import Model as km

class KerasModel(Model):
    
    def __init__(self,
                 TensorGraph : TensorGraph,
                 **kwargs):
        self.TensorGraph = TensorGraph
        self.model = None
        self.name = TensorGraph.name
        self.inputs = {k : Input(type_spec=v) for k,v in self.TensorGraph.inputs.items()}
        self.outputs = self.TensorGraph(self.inputs)
        
    def build(self,**kwargs):
        ''' builds and compiles the Keras model. Inputs and outputs
        are set as specified in TensorGraph.
        
        Parameters
        ----------
        
        kwargs : args passed as keras.Model.compile(kwargs)
        
        '''
        
        self.model = km(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(**kwargs)   
        
    def fit(self, dataset : Dataset | Batcher | StructuredDataset, workers=3, seed=None, **kwargs):
        """
        
        """
    
        #default options
        defaults = {'steps_per_epoch': 1,
                    'epochs'         : 1,
                    'batch_size'     : 1,
                    'shuffle'        : True
                   }
        
        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v
        
        #training batcher
        training_batcher = None
        if isinstance(dataset,Batcher):
            training_batcher = dataset
        
        if isinstance(dataset,(Dataset,StructuredDataset)):
            training_batcher = Batcher(dataset,kwargs['batch_size'],
                                       features=[list(self.inputs),list(self.outputs)],
                                       workers=workers,shuffle=shuffle,seed=seed)
        if not training_batcher:    
            raise ImproperModelError('training data must be either CREST batcher or dataset for training')
        
        #validation batcher
        validation_batcher = None
        if 'validation_data' in kwargs:
            if 'validation_steps' not in kwargs:
                kwargs['validation_steps'] = 1
            
            if isinstance(kwargs['validation_data'],Batcher):
                validation_batcher = kwargs['validation_data']
            
            if isinstance(kwargs['validation_data'],(Dataset,StructuredDataset)):
                validation_batcher = Batcher(kwargs['validation_data'],kwargs['batch_size'],
                                             features=[list(self.inputs),list(self.outputs)],workers=workers,shuffle=False,seed=seed)
                kwargs['validation_data'] = validation_batcher
            
            if not validation_batcher:    
                raise ImproperModelError('validation_data must be either a CREST batcher or dataset')
                
        self.model.fit(training_batcher,**kwargs)
        
    def predict(self, dataset : Dataset | Batcher | dict, workers=3, seed=None, **kwargs):
        
        #if dictionary
        if isinstance(dataset, dict):
            return self.model.predict(dataset,**kwargs)
       
        #default options since we use generators
        defaults = {
            'batch_size'  : 1,
            'steps'       : 1
        }
        
        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v
        
        if isinstance(dataset, Batcher):
            return self.model.predict(dataset,**kwargs)
        
        if isinstance(dataset,(Dataset,StructuredDataset)):
            batcher = Batcher(dataset,kwargs['batch_size'],
                                       features=[list(self.inputs),list(self.outputs)],
                              workers=workers,shuffle=False,seed=seed)
            
        return self.model.predict(batcher,**kwargs)

    def evaluate(self, dataset : Dataset | Batcher, workers=3, seed=None, **kwargs):
       
        #default options since we use generators
        defaults = {
            'batch_size'  : 1,
            'steps'       : 1
        }
        
        for k,v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v
        
        batcher = None
        if isinstance(dataset, Batcher):
            batcher = dataset
        
        if isinstance(dataset,(Dataset,StructuredDataset)):
            batcher = Batcher(dataset,kwargs['batch_size'],
                                       features=[list(self.inputs),list(self.outputs)],
                              workers=workers,shuffle=False,seed=seed)
            
        if isinstance(dataset,(tuple,list)):
            if len(dataset) != 2:
                raise ImproperModelError(f'dataset expected to be dim=2 found dim={len(dataset)}')
            return self.model.evaluate(dataset[0], dataset[1], **kwargs)
                
        if not batcher:
            raise ImproperModelError('This type is not supported for dataset')
            
        return self.model.evaluate(batcher, **kwargs)