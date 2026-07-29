""" Base class for model to constrain all inheriting classes """
from crest.base import BaseAbstract


class ImproperModelError(Exception):
    """ Raised when an improper Model is created """


class BaseModel(BaseAbstract):
    """ Model Base class """
