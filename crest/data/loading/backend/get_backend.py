from .TileDB import TileDB
from .Zarr import Zarr


def get_backend(path):
    """Return the correct backend object for the given database path.
    
    Parameters
    ----------
    path : str | Path | S3Path
        Location of the database that will be loaded.

    Returns
    -------
    BaseBackend
        The backend object used to load the given path.
    
    """
    
    # Extract the path extension
    ext = str(path).split('.')[-1].split('/')[0].split('\\')[0]

    if ext == 'zarr':
        return Zarr(path)
    if ext == 'tiledb':
        return TileDB(path)
    raise Exception(f'No backend available for loading "{path}": {ext}')