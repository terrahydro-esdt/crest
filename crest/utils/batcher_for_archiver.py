import tlz
import numpy as np
import tensorflow as tf

from crest.data.Batcher import Batcher
from crest.data.loading.Dataset import Dataset

# a function to get a coordinate value by indexing
# the coords of the samples in a batch
def _get_coordinate(sample,
                    coordinate_key: str,
                    index: int,):
 
    c = sample.coords[coordinate_key][index]
    c = c[-1] if c.size > 0 else c
 
    return c

# a function to get an index of a datafile in a dataset
# using the name of its zarr file source
def _get_index_from_name(df_name: str, dataset):
    if dataset is None:
        message = 'To capture the coordinate by the names'
        message += ' of the datafiles, specify the crest dataset'
        raise ValueError(message)
    
    try:
        return next(i for i, df in enumerate(dataset) if df.location.parent.name.lower() == df_name)
    except StopIteration:
        raise ValueError(f'{df_name} not found')
 
def batch_for_archiver(batcher: Batcher,
                       model_inputs: list[str],
                       coordinate_df: int | str | dict[str, int | str],
                       dataset: Dataset | None = None,
                       default_index: int | str = 0,):
    """A function to prepare the batches of samples for the archiver.
    
    The function gets a batch of samples from a crest batcher
    and captures the input sources and the corresponding coordinates
    to predict with a model and archive the prediction.
    
    Parameters
    ----------
    batcher         : crest.data.Batcher.Batcher
    A crest batcher. The features argument of the batcher must
    be None or this function does not work.
    
    model_inputs         : list
    The list of the features that the model needs for prediction
    
    coordinate_df         : int | str | dict
    The coordinates info that must be used for the samples. Each sample
    contains the coordinates information of all of the datafiles that
    participates in the batcher. if coordinate_df is of type int, it
    is considered as the index of the datafile in the dataset
    that the batcher uses. If of type str, it is considered as the name
    of the zarr file that the datafile were made from and the function
    captures the coordinate info of that. coordinate_df could also be a
    dict with the keys being the name of the coordinates and the values
    be the index of the datafile or the name of the zarr file of the
    datafile. If any coordinate is missing in the coordinate_df, the
    default_index is used for it.
    
    dataset         : crest.data.loading.Dataset.Dataset, optional
    The crest dataset that the batcher uses. It is only required
    if coordinate_df is of type str or a dict with str values.
    
    default_index         : int | str
    The default index for the coordinates that are missing in the
    coordinate_df dict. It can be either the index of the datafile
    in the dataset or the name of the zarr file of the datafile.
    

    Returns
    -------

    Raises
    ------
    ValueError: When the dataset is missing while the coordinate_df
    is of type str or a dict with str values.
    
    ValueError: When the datafile is not found using the name specified
    in coordinate_df.

"""

    # we need the first batch to make the coordinate_df if it is
    # an int or str
    batch = next(batcher)
    first_batch = True
    
    # prepare the coordinate_df for different cases
    if isinstance(coordinate_df, int):
        coordinate_df = {c: coordinate_df for c in batch[0].coords.keys()}

    if isinstance(coordinate_df, str):
        coordinate_df = {c: _get_index_from_name(coordinate_df, dataset) for c in batch[0].coords.keys()}
        
    if any(isinstance(v, str) for v in coordinate_df.values()):
        coordinate_df = {c: _get_index_from_name(v, dataset) for c, v in coordinate_df.items()}
        
    if isinstance(coordinate_df, dict):
        for k in batch[0].coords.keys():
            if not k in coordinate_df.keys():
                coordinate_df[k] = default_index if isinstance(default_index, int) else _get_index_from_name(default_index, dataset)
    
    # get rid of the features coordinate in the sample.coords
    coordinate_df.pop('features', None)
    
    # loop over the original batcher
    while True:
        try:
            batch = batch if first_batch else next(batcher)
            first_batch = False
        except StopIteration:
            break
        
        # get the coords
        coords = tlz.merge_with(np.array, [{c : _get_coordinate(s, c, coordinate_df[c]) for c in coordinate_df.keys()} for s in batch])
        # get the data
        x = tlz.merge_with(tf.constant, [s.to_dict(model_inputs) for s in batch])
 
        yield coords, x