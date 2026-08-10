from crest.base import BaseAbstract
import xarray as xr


class BaseBackend(BaseAbstract):
    """ Base class for storage backend modules """
    
    def __init__(self, path):
        self.path = path


    def open(self, path=None, **kwargs):
        """ Open the storage location and return an xarray object """
        raise Exception(f'Not implemented for {self}')


    def cache(self, path, data, stream=None):
        """ Write the given data to the storage location """
        raise Exception(f'Not implemented for {self}')