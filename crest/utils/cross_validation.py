import functools
import inspect
import numbers
import numpy as np
import xarray as xr
import dask.array as da

from crest.base.BaseAbstract import BaseAbstract
from crest.data.loading import Datafile

def _partial_to_function(partial_obj):
    def new_function(*args, **kwargs):
        # Combine the preset arguments from the partial with new arguments
        combined_args = partial_obj.args + args
        all_kwargs = {**partial_obj.keywords, **kwargs}
        return partial_obj.func(*combined_args, **all_kwargs)
    return new_function

class KfoldSplit(BaseAbstract):
    
    """Class which split a Datafile into train/test sets
       for k-fold cross-validation of a model.
       
       It creates a preprocessor for a crest Datafile that select 
       a region and return it as the test set. It also mask the
       selected region and return the rest for training.

    Parameters
    ----------
    folds      : int,
        The number of folds for splitting the data. The default
        is 5 which divides the Datafile into 5 folds.
    axis : list[str], Optional
        The names of the coordinates of the Datafile to be splitted.
    split_type  :   str
        The strategy for selecting pixels within folds. It can be
        either 'random' (default) or 'ordered'.
    target_var : list[str], Optional
        The variables of a Datafile which must be considered
        in splitting. The Splitter considers all
        the pixels with values (ignore the NaN pixels) and divides
        them into k folds. If a Datafile has multiple features,
        a list of them can be specified and the splitter take
        those features into account to creates the folds. If
        not specified, all features will be considered.
    fill_values : dict[str, list[float]], Optional
        A dictionary with the features as keys and a list of
        invalid pixels values to ignore by the splitter. If not
        specified, only NaN pixels are being ignored.
    seed         : int, Optional
        Seed for reproducible randomness. The random split must
        be controlled for the splitter because for requested fold,
        it shuffles the pixels and then returns fold. The shuffling
        step must be the same for all of the n folds requests.
        
    Examples
    --------
    >>> from crest.utils.kfold import KfoldSplit
    
    Create a random spatial 10 folds splitter for a Datafile
    
    >>> splitter = KfoldSplit(folds=10, axis=['latitude', 'longitude'])
    
    The function that returns first out of 10 folds split
    >>> train_first_split = splitter[1]['train']
    >>> test_first_split = splitter[1]['test']
    
    The function that returns second out of 10 folds split
    >>> train_second_split = splitter[2]['train']
    >>> test_second_split = splitter[2]['test']
    
    For a temporal test of a model, use date/time axis and ordered style.
    A Three years of data
    
    >>> splitter = KfoldSplit(folds=3, axis=['datetime'], split_type='ordered')
    
    These functions are used as the preprocessors of the test
    and train Datafiles

    """
    
    def __init__(self,
                 folds: int = 5,
                 axis: list[str] | None = None,
                 split_type: str = 'random',
                 target_var: list[str] | None = None,
                 fill_values: dict[str, list[numbers.Number]] | None = None,
                 seed: int = 0):

        self.folds = folds
        self.axis = axis
        self.split_type = split_type
        self.target_var = target_var
        self.fill_values = fill_values
        self.seed = seed
        
        
    
    def __getitem__(self, index):
        """A function to specify the fold that the splitter must return
           the train/test split for
        
        Parameters
        ----------
        index      : int,
            Either index of the selected fold or one of the
            ['train', 'test'].
        
        Raises
        ----------
        ValueError
            if for an n-fold cross validation, index is not in
            [1,n] or not equal to 'train'/'test'.
            
        Returns
        ----------
        KfoldSplit class | function
            the splitter class with the modified get_split  method
            or the get split method with k and/or split specified
        
        """
        
        # check if any partial function has been created
        get_split_args = inspect.getfullargspec(self.get_split).kwonlydefaults
        
        # if there is a partial function with both k and split
        # get back to the original get_split function.
        # This is necessary because multiple folds of a splitter
        # might be created by the user and this line makes sure
        # the last created fold does not overwrite the rest.
        if get_split_args and 'k' in get_split_args.keys() and 'split' in get_split_args.keys():
            self.get_split = self.get_split.func
        
        # check the partial function again
        get_split_args = inspect.getfullargspec(self.get_split).kwonlydefaults
        
        # for the case that index is the selected fold
        if isinstance(index, int):
            # if a partial function exists
            if get_split_args:
                # check if index of the selected fold is already used
                if 'k' in get_split_args.keys():
                    raise ValueError('Index must be of type str')

                # check if index is in the valid range
                if index not in range(1, self.folds + 1):
                    raise ValueError(f'index by fold must be in range {1, self.folds}')

                # create a partial version of the get_split function
                # with k specified and then convert the partial function
                # to a regular function. Datafile preprocessor cannot be
                # a partial function
                self.get_split = functools.partial(self.get_split, k=index)
                return _partial_to_function(self.get_split)

            # if a partial version of get_split does not exists
            # create one
            if index not in range(1, self.folds + 1):
                raise ValueError(f'index by fold must be in range {1, self.folds}')
        
            self.get_split = functools.partial(self.get_split, k=index)
            return self
        
        # for the case that index is either test or train
        elif isinstance(index, str):
            # if a partial function exists
            if get_split_args:
                # check if train/test is already used
                if 'split' in get_split_args.keys():
                    raise ValueError('Index must be of type int')

                # check if index is either train/test
                if index not in ['train', 'test']:
                    raise ValueError('index by split must be either train or test')

                # create a partial version of the get_split function
                # with split specified and then convert the partial function
                # to a regular function. Datafile preprocessor cannot be
                # a partial function
                self.get_split = functools.partial(self.get_split, split=index)
        
                return _partial_to_function(self.get_split)
            
            # if a partial version of get_split does not exists
            # create one
            if index not in ['train', 'test']:
                raise ValueError('index by split must be either train or test')
        
            self.get_split = functools.partial(self.get_split, split=index)
            return self
        
        #  
        else:
            raise ValueError('invalid index, must be either int or str')
    
    
    def _apply_split(self,
                     data_axis,
                     k,
                     split,):
        
        grid = data_axis.data
        
        # ravel the random_mask for easy shuffling
        # and get the pixels with valid values
        ref_shape = grid.shape
        grid = grid.ravel()
        i = da.stack(da.nonzero(grid),
                     allow_unknown_chunksizes=True,
                     axis=1).compute_chunk_sizes()

        # get the size and starting point of the
        # requested fold
        split_size = i.shape[0]//self.folds
        if split_size < 1:
            raise ValueError(f'Cannot split with fold size of {self.folds}')
        split_start = split_size * (k - 1)
        
        # randomly shuffle the ravelled mask if style is random
        if self.split_type == 'random':
            i = da.random.RandomState(seed=self.seed).permutation(i)
            
        # keep the test split and mask the rest
        if split == 'test':
            grid=da.zeros_like(grid)
            grid[i[split_start:split_start+split_size,0].compute()] = 1.
            
        
        # or mask the test split and keep the rest
        elif split == 'train':
            grid = grid/grid
            grid[i[split_start:split_start+split_size,0].compute()] = 0.
            
        else:
            raise ValueError('split must be either train or test')

        # reshape the random mask and update the masking
        # grid by that
        grid = da.reshape(grid, shape=ref_shape)
        grid = xr.DataArray(grid, dims=data_axis.coords)
        grid = grid.where(grid != 0.)
        data_axis = data_axis * grid
            
        return data_axis
        
      
    def get_split(self,
                  datafile : Datafile,
                  data : xr.Dataset,
                  split : str,
                  k     : int,
                  ):
        
        """The function that prepares the train/test splits for
           a crest Datafile
           
           This function has been designed in accordance with the
           structure mandated by the Crest Datafile preprocessor.
        
        Parameters
        ----------
        datafile      : Datafile,
            A crest Datafile
        data          : xarray Dataset,
            The data source of the datafile
        split         : str
            The test or train split selector
        k             : int
            index of the selected fold. must be
            in range [1, folds]
        
        Returns
        ----------
        xr.Dataset
            The data source of the datafile modified to be a train/test
            split
        
        Raises
        ----------
        ValueError
            If any one of the provided axis is not found
            among the coordinates of the data source of the datafile.
        ValueError
            If the selected split strategy is not 'random' or
            'ordered'.
        ValueError
            If any one of the variables in fill_values is not found
            among the variables of the data source of the datafile.
        ValueError
            If any one of the variables in target_var is not found
            among the variables of the data source of the datafile.
        
        """
        # select the features of the data source based on the datafile
        # settings
        if datafile.features:
            data = data[datafile.features]
        
        # check if all requested axis exist in the data source
        coords = data.coords
        if self.axis:
            if not set(self.axis).issubset(coords):
                raise ValueError('invalid selected axis')
            
        # check split_type
        if self.split_type not in ['random', 'ordered']:
            raise ValueError('invalid split_type, choose either random or ordered')
        
        # select an extent of the data source based on the datafile settings
        data = data.sel({c: slice(*ext) for c, ext in datafile.extent.items() if c in coords})
        
        # ignore the requested fill_values for splitting
        if self.fill_values:
            for key in self.fill_values.keys():
                if key not in data.keys():
                    raise ValueError(f'invalid variable in fill_values: {key}')
                data[key] = data[key].where(~np.isin(data[key], self.fill_values[key]))
        
        # get the axis that are not selected to eliminate them from the splitter
        if self.axis:
            count_over_dims = [d for d in coords if d not in self.axis + ['features']]
            data_axis = data.count(count_over_dims) if len(count_over_dims) > 0 else data.where(np.isnan(data), 1.)
            data_axis = data_axis/data_axis
        else:
            data_axis = data.where(np.isnan(data), 1.)

        # extract the selected variables (based on the target_vars) and
        # ignore the rest
        if self.target_var:
            if not set(self.target_var).issubset(data.keys()):
                raise ValueError('invalid variable(s) in target_var')
            data_axis = data_axis[self.target_var]
        
        # combine the valid pixels from all features of the datafile
        data_axis = data_axis.to_array('features').sum('features')

        # apply split
        data_axis = self._apply_split(data_axis=data_axis,
                                        k = k,
                                        split=split,)
        
        # and finally apply the masking grid to the data
        data = data * data_axis
        return data
    