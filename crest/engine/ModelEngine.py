from __future__ import annotations
import os
import logging
import traceback

from crest.model.GriddedModel import GriddedModel
from crest.configuration.Config import Config
from crest.data.loading.Dataset import Dataset
from crest.engine.Engine import Engine

log = logging.getLogger(__name__)

class ModelEngine(Engine):
    """ CREST Engine performs an update on a specified model.
        The a path to the model archive must be provided within the config file.
        The update of the model is performed with the data specified in the config. 
    """

    def __init__(self, config: str | Config, **kwargs):
        """ Initializes the ModelEngine. 

        Parameters
        ----------
        config: str
            Path to the YAML format configuration file that helps to configure the digital
            replica engine and the gridded model. 
        """
        log.info(f'Initializing ModelEngine')

        log.info(f'Load configuration.')
        if (isinstance(config, str)):
            self.config = Config(config)
        else:
            self.config = config

        self.model_loader = kwargs.get('model_loader', None)
        self.database_path = kwargs.get('database_path', None)

        self.process_dataset = kwargs.get('process_dataset', {
        'preprocess': self.preprocess_dataset,
        'postprocess': self.postprocess_dataset,
        })

        self.process_model = kwargs.get('process_model', {
            'preprocess': self.preprocess_model,
            'postprocess': self.postprocess_model,
        })

        self.process_output = kwargs.get('process_output', {
            'postprocess': self.postprocess_output,
        })

        self.data_schema_adapter = kwargs.get(
            'data_schema_adapter', self.data_schema_adapter)
        
        self.benchmarking = kwargs.get('benchmarking', {
            'benchmark_tdt': self.benchmark_targetdt,
        })

        log.info(
            'Create the directory required to store all out-going data.')

        output_path = self.config.archive_kwargs['output_path']
        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'logs'))

        log.info(f'Initialize Gridded Model to initiate an update.')
        self.gridModel = GriddedModel(
            self.config, 
            database_path=self.database_path,
            alt_model_loader=self.model_loader, 
            process_model=self.process_model, 
            process_output=self.process_output,
            benchmarking=self.benchmarking)

    def reset_config(self, new_config: Config | str):
        """ Set self.config to different configuration. 
            Reinitializes gridded model with the new configuration. 

        Parameters
        ----------
        new_config: CREST Config type object or str
             Resets the configuration previously specified with either the CREST config 
             object or with the path to the config. 

        Returns
        -------
        boolean
            True if the configuration was successfully set, False otherwise. 
        """

        try:
            if (isinstance(new_config, str)):
                self.config = Config(new_config)
            else:
                self.config = new_config

            self.gridModel = GriddedModel(self.config)
            return True
        except Exception as e:
            log.exception(f'Could not set configuration for DRE. {e=}')
            return False

    def update_config(self, new_config: Config | str | dict):
        """ Update the configuration. Integrates new settings and overwrites the configurations
            that were previously set. Very similiar to reset_config but allows for a subtle 
            reshaping of configuration rather than a full rewrite. 

        Parameters
        ----------
        new_config: Config | str | dict
            The underlying config object is a dictionary. The new setting are loaded and merged
            with dict.update. 

        Returns
        -------
        boolean:
            True if the merge was successful and False otherwise.
        """

        log.info('Updating the config with input.')
        try:
            nc = None
            if (isinstance(new_config, str)):
                nc = Config(new_config)
                nc = nc.config
            elif (isinstance(new_config, Config)):
                nc = new_config.config
            else:
                nc = new_config

            log.info('Update the underlying config dictionary.')
            self.config.update(nc)

            log.info(f'Reinitialize Gridded Model.')
            self.gridModel = GriddedModel(self.config)

            return True
        except Exception as e:
            log.exception(f'Could not update config: {e}')
            return False

    def cast(self):
        """ Updating the model based on data specified in config. """

        try:
            log.info(f'Update dataset.')
            data = self.gridModel.init_dataset(
                process_dataset=self.process_dataset)

            if self.data_schema_adapter is not None:
                log.info(f'Adapting dataset schema.')
                data_schema = self.data_schema_adapter(data)
            else:
                data_schema = data

            log.info(f'Update ModelEngine')
            return self.gridModel.predict(data_schema)
        except Exception as e:
            traceback.print_stack()
            log.exception(f'Could not complete casting. {e}')
            return False
        
    def benchmark_targetdt(self, data: Dataset, database_path: str):
        pass
    
    def preprocess_dataset(self, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} must implement preprocess_dataset')

    def postprocess_dataset(self, ds: Dataset, **kwargs):
        return ds
    
    def preprocess_model(self, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} must implement preprocess_model')
    
    def postprocess_model(self, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} must implement postprocess_model')
    
    def postprocess_output(self, **kwargs):
        raise NotImplementedError(f'{type(self).__name__} must implement postprocess_output')

    def data_schema_adapter(self, data: Dataset):
        raise NotImplementedError(f'{type(self).__name__} must implement data_schema_adapter')