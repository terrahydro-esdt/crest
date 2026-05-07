from crest.base import BaseAbstract
import xarray as xr


class BaseBackend(BaseAbstract):

    def __init__(self, path):
        self.path = path


    def open(self, path=None, **kwargs):
        raise Exception(f'Not implemented for {self}')


    def cache(self, path, data, stream=None):
        raise Exception(f'Not implemented for {self}')