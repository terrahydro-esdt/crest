"""
This module implements a configuration loader based on reading YAML files
"""


import yaml
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    """ CREST Configuration class handles loading and saving CREST workflow configuration settings
        from a YAML file. 
    """

    def __init__(self, config_file: str = None):
        self.config_file = config_file

        logger.info('Initializing Config class')

        self._config = {}
        if (config_file and os.path.exists(config_file)):
            self._config = self.load()

        if (config_file is None):
            logger.debug(f'No config passed on initialization of Crest Config.')

    def load(self, config_file: str = None):
        """ Safetly loads the YAML configuration file. """

        try:
            if (config_file):
                self.config_file = config_file

            logger.info(f'Loading configurations based on file {self.config_file}')

            with open(self.config_file, 'r') as file:
                self._config = yaml.safe_load(file)

            logger.info(f'Configurations loaded {self._config}')

            return self._config
        except Exception as e:
            logger.exception(f'Could not safetly load YAML. {e}')
            raise e

    def save(self, filepath: str = None):
        """ Saves the current configuration dictionary as YAML to either the original file 
            or the filepath specified. 

        Parameters
        ----------
        filepath: str
            The string representation of the filepath where the user wished to save the dictionary of 
            configurations set in current object.     
        """
        if (not filepath):
            logger.info(
                f'filepath is not specified, save() will overwrite original config file {self.config_file}')
            filepath = self.config_file

        if (not filepath):
            logger.exception(f'no filepaths are available, aborting save()')
            raise ValueError(f'Could not save to filepath {filepath}')

        try:
            with open(filepath, 'w') as file:
                yaml.dump(self._config, file, default_flow_style=False)

            logger.info(f'Configuration saved to {filepath}')
        except Exception as e:
            logger.exception(f'Failed to save configuration: {e}')
            raise e

    @property
    def config(self):
        """ Dictionary of loaded configruation parameters """
        return self._config

    def update(self, to_update: dict):
        """ Updates the underlying config dictionary with new dictionary specified. """
        return self._config.update(to_update)

    def __getattr__(self, name):
        """ Forwards attribute access to the underlying config dict. """

        if name in self._config:
            return self._config[name]
        
        return None
    
    def __str__(self):
        """ String representation of configuration dictionary """
        return f'Configuration: {self._config}'

    def __repr__(self):
        """ Show configuration dictionary """
        return f'Config({self.config_file})'
