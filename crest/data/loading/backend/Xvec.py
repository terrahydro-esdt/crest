from .BaseBackend import BaseBackend

from dask.diagnostics import ProgressBar
import xarray as xr
import sys
try:
    import xvec
except ImportError:
    xvec = None


class Xvec(BaseBackend):
    """Backend for xvec-encoded Zarr storage."""

    def open(self, path=None, **kwargs):
        if xvec is None:
            raise ImportError('Install xvec package to use the xvec backend.')
        if path is None:
            path = self.path

        data = xr.open_zarr(path, **kwargs)
        if 'geometry_container' in data.data_vars:
            data = data.xvec.decode_cf()
        return data

    def cache(self, dest, data, stream=sys.stdout, **kwargs):
        if xvec is None:
            raise ImportError('Install xvec package to use the xvec backend.')
        if 'geometry' in data.coords: 
            data = data.xvec.encode_cf()
        with ProgressBar(out=stream):
            data.to_zarr(dest, **kwargs)
