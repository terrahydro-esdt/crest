""" Base class for TensorGraphs """

from ..base import BaseAbstract


class ImproperTensorGraphError(Exception):
    """ Raised when an improper TensorGraph is created """

class TensorGraph(BaseAbstract):
    """ Non-hierarchal basecalss"""
