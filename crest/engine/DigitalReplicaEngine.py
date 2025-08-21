from crest.model.GriddedModel import GriddedModel
from crest.configuration.Config import Config
import os
import logging

logger = logging.getLogger(__name__)


class DigitalReplicaEngine():
    """ Digital Replica Engine performs an update on a specified model.
        The a path to the model archive must be provided within the config file.
        The update of the model is performed with the data specified in the config. 
    """

    def __init__(self, config: str, model_loader=None, process_dataset=None):
        """ Initializes the DigitalReplicaEngine. 

        Parameters
        ----------
        config: str
            Path to the YAML format configuration file that helps to configure the digital
            replica engine and the gridded model. 
        """
        logger.info(f'Initializing DigitalReplicaEngine')

        logger.info(f'Load configuration.')
        self.config = Config(config)

        self.model_loader = model_loader
        self.process_dataset = process_dataset

        logger.info(
            'Create the directory required to store all out-going data.')
        if not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)
            os.makedirs(os.path.join(self.config.output_path, 'logs'))

        logger.info(f'Initialize Gridded Model to initiate an update.')
        self.gridModel = GriddedModel(self.config, alt_model_loader=model_loader)

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
            logger.exception(f'Could not set configuration for DRE. {e=}')
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

        logger.info('Updating the config with input.')
        try:
            nc = None
            if (isinstance(new_config, str)):
                nc = Config(new_config)
                nc = nc.config
            elif (isinstance(new_config, Config)):
                nc = new_config.config
            else:
                nc = new_config

            logger.info('Update the underlying config dictionary.')
            self.config.update(nc)

            logger.info(f'Reinitialize Gridded Model.')
            self.gridModel = GriddedModel(self.config)

            return True
        except Exception as e:
            logger.exception(f'Could not update config: {e}')
            return False

    def nowcast(self):
        """ Updating the model based on data specified in config. """

        try:
            logging.info(f'Update dataset.')
            self.gridModel.init_dataset(process_dataset=self.process_dataset)

            logging.info(f'Update DigitalReplicaEngine')
            return self.gridModel.predict()
        except:
            logger.exception('Could not complete nowcast.')
            return False
