from crest.utils import S3Path
from .BaseBackend import BaseBackend

from dask.diagnostics import ProgressBar
from fsspec.mapping import FSMap
import xarray as xr
import zarr
import sys


class Zarr(BaseBackend):
    """ Backend class to handle reading and writing to Zarr storage """
    
    def open(self, path=None, **kwargs):
        """ Open a Zarr location and return the associated xarray object """
        if path is None:
            path = self.path
        
        # if not isinstance(path, (S3Path, FSMap)):
        #     path = zarr.DirectoryStore(path)
        return xr.open_zarr(path, **kwargs)


    def cache(self, dest, data: xr.Dataset, stream=sys.stdout, **kwargs):
        """ Write the given xarray Dataset to a Zarr database """
        with ProgressBar(out=stream): 
            data.to_zarr(dest, **kwargs)