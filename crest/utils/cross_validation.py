import numpy as np
import xarray as xr
import dask.array as da

from crest.base.BaseAbstract import BaseAbstract
from crest.data.loading import Datafile

class SpatialKfoldRandomSplit(BaseAbstract):
    
    """Class which split a Datafile into train/test sets spatially and
       randomly for k-fold cross-validation of a model.
       
       It creates a preprocessor for a crest Datafile that select 
       a region and return it as the test set. It also mask the
       selected region and return the rest for training.

    Parameters
    ----------
    folds      : int,
        The number of folds for splitting the data. The default
        is 5 which divides the Datafile into 5 folds spatially.
    target_var : list[str], Optional
        The variables of a Datafile which must be considered
        for the spatial splitting. The Splitter considers all
        the pixels with values (ignore the NaN pixels) and divides
        them into k folds. If a Datafile has multiple features,
        a list of them can be specified and the splitter take
        those features into account to creates the folds. If
        not specified, all features will be considered.
    fill_values : dict[str, list[float]], Optional
        A dictionary with the features as keys and a list of
        invalid pixels values to ignore by the splitter. If not
        specified, only NaN pixels are being ignored.
    spatial_dims : list[str], Optional
        The names of the spatial coordinates of the Datafile.
        Default is ['latitude', 'longitude'].
    seed         : int, Optional
        Seed for reproducible randomness. The random split must
        be controlled for the splitter because for requested fold,
        it shuffles the pixels and then returns fold. The shuffling
        step must be the same for all of the n folds requests.
        
    Examples
    --------
    >>> from crest.utils.kfold import SpatialKfoldRandomSplit
    
    Create a 10 fold splitter for a Datafile
    
    >>> splitter = SpatialKfoldRandomSplit(folds=10)
    
    The function that returns first out of 10 folds split
    >>> train_first_split = splitter[1].train
    >>> test_first_split = splitter[1].test
    
    The function that returns second out of 10 folds split
    >>> train_second_split = splitter[2].train
    >>> test_second_split = splitter[2].test
    
    These functions are used as the preprocessors of the test
    and train Datafiles

    """
    
    def __init__(self,
                 folds: int = 5,
                 target_var: list[str] | None = None,
                 fill_values: dict[str, list[float]] | None = None,
                 spatial_dims: list[str] = ['latitude', 'longitude'],
                 seed: int = 0):

        self.target_var = target_var
        self.fill_values = fill_values
        self.folds = folds
        self.spatial_dims = spatial_dims
        self.seed = seed

        self.k = None

        
    def __getitem__(self, k):
        """A function to specify the fold that the splitter must return
           the train/test split for
        
        Parameters
        ----------
        k      : int,
            The index of the fold.
        
        Raises
        ----------
        ValueError
            if for an n-fold cross validation, k is less than
            1 or more than n.
            
        Returns
        ----------
        Self
            the splitter class with the modified k attribute
        
        """
        self.k = k
        
        if self.k not in range(1, self.folds + 1):
            raise ValueError(f'k_index must be in range {1, self.folds}')
            
        return self
    
    @property
    def train(self):
        return self._get_train_split
    
    @property
    def test(self):
        return self._get_test_split
    
    def _get_train_split(self,
                        datafile,
                        data,):
        return self.get_split(datafile, data, 'train')
    
    def _get_test_split(self,
                        datafile,
                        data,):
        return self.get_split(datafile, data, 'test')
        
    def get_split(self,
                  datafile : Datafile,
                  data : xr.Dataset,
                  split : str,
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
        
        Returns
        ----------
        xr.Dataset
            The data source of the datafile modified to be a train/test
            split
        
        Raises
        ----------
        ValueError
            if any one of the provided spatial_dims is not found
            among the coordinates of the data source of the datafile.
            
        ValueError
            if split is not train or test.
        
        """
        # select the features of the data source based on the datafile
        # settings
        data = data[datafile.features]
        
        # check if all requested spatial_dims exist in the data source
        coords = data.coords
        invalid_keys = [c for c in self.spatial_dims if c not in coords]
        if len(invalid_keys) > 0:
            raise ValueError(f'invalid keys: {invalid_keys}')
        
        # select an extent of the data source based on the datafile settings
        data = data.sel({c: slice(*ext) for c, ext in datafile.extent.items() if c in coords})
        
        # ignore the requested fill_values for splitting
        if self.fill_values:
            for key in self.fill_values.keys():
                data[key] = data[key].where(~np.isin(data[key], self.fill_values[key]))
        
        # get the non-spatial coordinates to eliminate them from the splitter
        count_over_dims = [d for d in coords if d not in self.spatial_dims + ['features']]
        
        # if any non-spatial coordinates, aggregate over them by counting valid pixels
        if len(count_over_dims) > 0:
            data_spatial = data.count(count_over_dims)
        # else, just replace the values of the valid pixels by 1.
        # for the sake of consistency
        else:
            data_spatial = data.where(np.isnan(data), 1.)
        
        # extract the selected variables (based on the target_vars) and
        # ignore the rest
        if self.target_var:
            data_spatial = data_spatial[self.target_var]
        
        # combine the valid pixels from all features of the datafile
        # and get a 2D data
        data_spatial = data_spatial.to_array('features').sum('features')

        # shuffle and split the valid pixels of the 2D data randomly
        grid = data_spatial.data.ravel()
        i = da.stack(da.nonzero(grid),
                    allow_unknown_chunksizes=True,
                    axis=1).compute_chunk_sizes()
        
        split_size = i.shape[0]//self.folds
        split_start = split_size * (self.k - 1)
        
        rnd_ind = da.random.RandomState(seed=self.seed).permutation(i)[split_start:split_start+split_size]

        # keep the test split and mask the rest
        if split == 'test':
            grid=da.zeros_like(grid)
            grid[rnd_ind[:,0].compute()] = 1.
            
        
        # or mask the test split and keep the rest
        elif split == 'train':
            grid = grid/grid
            grid[rnd_ind[:,0].compute()] = 0.
            
        else:
            raise ValueError('split must be either train or test')

        grid = da.reshape(grid, shape=data_spatial.shape)
        data_spatial = xr.DataArray(grid, dims=self.spatial_dims)
        data_spatial = data_spatial.where(data_spatial != 0., np.nan)

        data = data * data_spatial
        
        return data
    