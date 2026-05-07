from crest.utils import S3Path
from .BaseBackend import BaseBackend

from dask.diagnostics import ProgressBar
from fsspec.mapping import FSMap
import xarray as xr
import zarr
import sys


class Zarr(BaseBackend):
    
    def open(self, path=None, **kwargs):
        if path is None:
            path = self.path
        
        # if not isinstance(path, (S3Path, FSMap)):
        #     path = zarr.DirectoryStore(path)
        return xr.open_zarr(path, **kwargs)


    def cache(self, dest, data: xr.Dataset, stream=sys.stdout, **kwargs):
        with ProgressBar(out=stream): 
            data.to_zarr(dest, **kwargs)