from crest.model.GriddedModel import GriddedModel
from crest.configuration.Config import Config
import os
import logging

log = logging.getLogger(__name__)


class DigitalReplicaEngine():
    """ Digital Replica Engine performs an update on a specified model.
        The a path to the model archive must be provided within the config file.
        The update of the model is performed with the data specified in the config. 
    """

    def __init__(self, config: str | Config, **kwargs):
        """ Initializes the DigitalReplicaEngine. 

        Parameters
        ----------
        config: str
            Path to the YAML format configuration file that helps to configure the digital
            replica engine and the gridded model. 
        """
        log.info(f'Initializing DigitalReplicaEngine')

        log.info(f'Load configuration.')
        if (isinstance(config, str)):
            self.config = Config(config)
        else:
            self.config = config

        self.model_loader = kwargs.get('model_loader', None)
        self.process_dataset = kwargs.get('process_dataset', None)
        self.process_model = kwargs.get('process_model', None)
        self.data_schema_adapter = kwargs.get('data_schema_adapter', None)

        log.info(
            'Create the directory required to store all out-going data.')

        output_path = self.config.archive_kwargs['output_path']
        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, 'logs'))

        log.info(f'Initialize Gridded Model to initiate an update.')
        self.gridModel = GriddedModel(
            self.config, alt_model_loader=self.model_loader, process_model=self.process_model)

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

            log.info(f'Update DigitalReplicaEngine')
            return self.gridModel.predict(data_schema)
        except:
            log.exception('Could not complete casting.')
            return False
