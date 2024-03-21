import json
import logging
import logging.config
import pathlib

from crest import ROOT_PATH


def logger_setup(config_name):
    """Set up the CREST library logger.

    Parameters
    ----------
    config_name (str) : configuration name prefix (as defined in utils/logging_configs/).
    """
    config_file = pathlib.Path(
        ROOT_PATH) / pathlib.Path("utils/crest_logging_configs/"+config_name+".json")
    
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
