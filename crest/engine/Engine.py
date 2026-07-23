"""
This module implements the base Engine class to be used to
define required functionality for sub-classing
"""

from crest.base import BaseAbstract


class ImproperEngineError(Exception):
    """ Raised when an improper Engine is created """
    pass


class Engine(BaseAbstract):
    """ Base Engine class """
    pass
