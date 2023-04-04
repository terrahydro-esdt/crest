from collections import defaultdict as dd
try:    from IPython.display import display
except: pass

import xarray as xr 
import numpy as np 
import sys 


def create_array(
    dimensions  : dict, 
    chunks      : dict  = {}, 
    missing_pct : float = 0, 
    random      : bool  = False,
    random_seed : int   = None,
) -> xr.DataArray:
    """ Create a DataArray

    Parameters
    ----------
    dimensions  : dict
        Dict mapping {coordinate: shape} for the DataArray.
    chunks      : dict
        Chunks that the data should be divided into.
    missing_pct : float
        Percentage of values that should be NaN.
    random      : bool
        Whether to use random floats, or just a monotonic range
        of numbers from [0, size of DataArray].
    random_seed : int
        Seed for the random generator.

    Returns
    -------
    xr.DataArray
        DataArray created using the requested parameters.

    """
    scale_num = dd(lambda: 0.1, time=10.)
    coord_num = lambda k, size: np.arange(size, dtype='float32')/scale_num[k]
    coord_str = lambda k, size: [f'var_{i}' for i in range(size)]
    gen_coord = lambda k, size: (coord_num, coord_str)[k=='features'](k, size)
    dimensions.setdefault('features', 1)

    numpy_rng    = np.random.default_rng(random_seed)
    keys, sizes  = zip(*dimensions.items())
    if random: a = numpy_rng.random(sizes, dtype='float32')
    else:      a = np.arange(np.prod(sizes), dtype='float32').reshape(sizes)
    missing_idxs = i = numpy_rng.integers(0, a.size, int(a.size * missing_pct))
    a.ravel()[i] = np.nan 
    coordinates  = dict(zip(keys, map(gen_coord, keys, sizes)))
    valid_chunks = {k:v for k,v in chunks.items() if k in keys}
    return xr.DataArray(a, coords=coordinates).chunk(valid_chunks)


def align(data: list[xr.DataArray]) -> list[xr.DataArray]:
    """ Modify coordinates such that all arrays fall within the same bounds 
    
    Parameters
    ----------
    data : list[xr.DataArray]
        List of DataArrays to align with one another.

    Returns
    -------
    list[xr.DataArray]
        DataArrays that were passed as input, but with their coordinates
        shifted and stretched so that all DataArrays are within the same
        coordinate bounds.

    """
    for c in data[0].dims:
        valid = [array for array in data if c in array.dims]
        minim = min(array[c].min() for array in valid)
        maxim = max(array[c].max() for array in valid)
        if minim.dtype.kind == 'O': continue 

        for i, array in enumerate(data):
            if c in array.dims:
                shift   = array[c].min() - minim 
                scale   = maxim // (array[c].max() - shift)
                updated = ((array[c] - shift) * scale).astype('float32')
                data[i] = array.assign_coords({c: updated})
    return data 


def examine(data: xr.DataArray) -> None:
    """ Print (or display, if in a jupyter notebook) the given DataArray
    
    Parameters
    ----------
    data : xr.DataArray
        DataArray to examine.

    """
    in_notebook = 'ipykernel' in sys.modules 
    divider     = '\n' + ''.join(['-']*80) + '\n'
    if in_notebook: display(data.to_dataset('features'))
    else:           print(data.transpose('features', ...), divider)


def synthetic_data(
    data_kwargs : list[dict], 
    verbose     : bool = True, 
    **universal
) -> list[xr.Dataset]:
    """ Generates a list of xr.Dataset objects containing synthetic data. 

    Combines the above methods into a single helper function. See configs.py
    for example configurations. 
    
    Parameters
    ----------
    data_kwargs : list[dict]
        List of kwargs dictionaries used to create the data. These kwargs 
        are passed into create_array, and so that function definition should
        be referenced to determine the contents of each dictionary.
    verbose     : bool = True
        Determines whether or not the created data objects should be printed
        or displayed by the examine function. 
    universal
        Any keyword arguments which should be used universally, across all
        of the generated data objects. These are used to update the data_kwargs
        dictionaries, and so are passed into the create_array function.

    Returns
    -------
    list[xr.Dataset]
        A list of the generated synthetic xr.Dataset objects.
        
    """
    data = [create_array(**(conf|universal)) for conf in data_kwargs]
    data = align(data)
    if verbose: list(map(examine, data))
    return [d.to_dataset('features') for d in data]