from crest.base import BaseAbstract


class ImproperTensorGraphError(Exception):
    """ Raised when an improper TensorGraph is created """
    pass


class TensorGraph(BaseAbstract): pass
