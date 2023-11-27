from scipy.spatial import KDTree
from pathlib import Path

import dask.array as da
import xarray as xr
import pandas as pd
import numpy as np
import typing

from crest.base import BaseAbstract
from crest.data.loading import Dataset, Datafile


class Archiver(BaseAbstract):

    """Class which handles inserting the model predictions into the
       corresponding coordinates of an xarray dataset and write the
       result into disk.

    Parameters
    ----------
    output_path         : str | Path,

        This argument specifies the path to the output file, which can be
        in either zarr ('.zarr') or netcdf ('.nc') formats. The class
        will automatically determine the file format based on the provided
        path. If no file extension is given, it will default to using the
        zarr format. The dataset located at the output_path (if exists) is
        treated as the default data_schema for storing predicted
        values and it must have at least one variable.

    data_schema      : Datafile | Dataset | None, Optional
        This argument pertains to a Datafile used for storing predicted
        values. The user has the option to provide either a single Datafile
        or a Dataset, which is essentially a list of Datafiles. If a
        Dataset is used, the user must also indicate which specific
        datafile to use by specifying the datafile_index, or else an
        error will be raised. The predicted values are then inserted
        into the nearest coordinate within the chosen Datafile.
        If the output_path exists, the specified data_schema is ignored.

    datafile_index      : int | None, Optional
        The index of the Datafile to be used as the data_schema if a
        Dataset is specified as data_schema. Default is the Datafile
        with the finest resolution.

    aggregate_func      : typing.Callable, Optional
        The function to be used as the aggregator of the predicted values
        of the exact same coordinate. For instance, when the predicted
        values are in a finer resolution than the data_schema, several
        values must be assigned to a single coordinate.
        Those values are aggregated by the aggregate_func. The
        function must be specified with a mandatory argument named axis=0.
        This is for the pandas.core.groupby.DataFrameGroupBy.apply() method.
        The default is the average function: lambda x: np.mean(x, axis=0).

    exact_coord_match   : bool, Optional
        If True, the coordinates of the predicted values are matched with
        the data_schema coordinates with KDTree method of the scipy package
        (docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html).
        It needs more memory and time to be executed. If False (Default), the
        inexact "nearest" method is used for the coordinate matching
        (docs.xarray.dev/en/latest/user-guide/indexing.html#nearest
        -neighbor-lookups).

    **kwargs
        Any additional kwargs are passed into either xr.Dataset.to_zarr or
        xr.Dataset.to_netcdf according to the suffix from output_path.

    Raises
    ------
    ValueError
        - If the dataset located at the output_path has no variables.
        - If output_path does not exist and the data_schema is not specified.
        - If datafile_index is out of range.

    """
    def __init__(self,
                 output_path: str | Path,
                 data_schema: Datafile | Dataset | None = None,
                 datafile_index: int | None = None,
                 aggregate_func: typing.Callable = lambda x: np.mean(x, axis=0),
                 exact_coord_match: bool = False,
                 **kwargs
                 ):

        self.output_path = output_path
        self.data_schema = data_schema
        self.datafile_index = datafile_index
        self.aggregate_func = aggregate_func
        self.exact_coord_match = exact_coord_match
        self.kwargs = kwargs

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # set the default suffix of zarr for the output file
        if self.output_path.suffix.lower() == '':
            par = self.output_path.parent
            nam = self.output_path.name
            self.output_path = par / (nam + '.zarr')

        # read the dataset at the output_dir as the data_schema
        if self.output_path.exists():
            self.out_datafile = self._open_file(output_dir=self.output_path)
            if len(list(self.out_datafile.keys())) == 0:
                message = f'The dataset at {self.output_path} must have '
                message += 'at list one variable'
                raise ValueError(message)

        # Use the specified data_schema if the output_dir does not exist
        else:
            if not self.data_schema:
                message = 'data_schema must be specified if '
                message += 'output_path does not exist'
                raise ValueError(message)

            # if the data_schema is of type Dataset
            if isinstance(self.data_schema, Dataset):

                # By default, the Datafile with the finest resolution
                # is considered as the data_schema unless it is defined
                # by the user
                if not self.datafile_index:
                    self.datafile_index = np.argmin(
                        [np.prod(df.resolution) for df in self.data_schema])

                else:
                    if self.datafile_index < 0 or self.datafile_index >= len(self.data_schema):
                        message = 'datafile_index is out of range. '
                        message += f'The valid range is '
                        message += f'(0, {len(self.data_schema)-1})'
                        raise ValueError(message)

                self.data_schema = self.data_schema[self.datafile_index]

            # convert the features coordinate to separate variables
            self.data_schema = self.data_schema.data.to_dataset(
                dim='features')

            # create the empty output dataset using the schema and save it to
            # the output_dir
            self.out_datafile = xr.full_like(self.data_schema,
                                             np.nan).drop_vars(list(self.data_schema.keys()))
            message = f'initializing {self.output_path.name} at '
            message += f'{str(self.output_path.parent)} ...'
            print(message)
            self._to_file(dataset=self.out_datafile,
                          output_dir=self.output_path,
                          mode='w')
            print('Done')

        # save the list of coordinates for future uses
        self.coords = list(self.out_datafile.coords)

        # create the KDTree if the exact coord matching is asked
        if self.exact_coord_match:
            self.lookup_space = self._create_coord_search_space()
            self.s = KDTree(self.lookup_space)

    # a function to create the combination of all coordinate values
    def _create_coord_search_space(self):

        args = (self.out_datafile[c] for c in self.coords)
        search_space = np.vstack(np.meshgrid(
            *args)).reshape(len(self.coords), -1).T

        return search_space

    # a function for coordinate matching
    def _match_coords(self,
                      pred_data: dict[str, np.ndarray]
                      ) -> dict[str, np.ndarray]:

        # get the coordinate keys and values out of the predictions
        # dictionary
        data_dict = {k: pred_data[k] for k in self.coords}

        # exact coordinate matching by KDTree if it is asked
        if self.exact_coord_match:
            query = np.array(list(data_dict.values())).T
            query = np.squeeze(self.s.query(query)[1])
            query = self.lookup_space[query]
            nearest = {k: query[:, i] for i, k in enumerate(data_dict.keys())}

        # inexact coordinate matching
        else:
            nearest = dict(self.out_datafile.sel(data_dict,
                                                 method='nearest'
                                                 ).coords.variables
                           )
            nearest = {i: v.to_numpy() for i, v in nearest.items()}

        # restore the non-coordinate keys and values into the predictions
        # dictionary
        not_coord_keys = {k: v for k, v in pred_data.items()
                          if k not in list(nearest.keys())}

        return nearest | not_coord_keys

    # a function to aggregate the predicted values that
    # are for a same coordinate
    def _aggregate(self,
                   pred_data: dict[str, np.ndarray]
                   ) -> pd.DataFrame:

        # make a pandas dataframe out of the predictions dict
        pred_df = pd.DataFrame.from_dict(pred_data)

        # group the same rows and aggregate them by the specified
        # function
        pred_df = pred_df.groupby(self.coords)
        pred_df = pred_df.apply(func=lambda x: self.aggregate_func(x))

        return pred_df

    # a function to insert values into the correct coordinate of
    # the output xarray dataset
    def _insert_values(self,
                       pred_df: pd.DataFrame,
                       dataset: xr.Dataset):

        # find the non-coordinate keys (variables) in the predictions dict
        vars = [col for col in pred_df.columns if col not in self.coords]

        # vectorize the process of inserting the predicted variables
        # into the corresponding coordinates
        for var in vars:
            def f(coord, val): dataset[var].loc[coord] = val
            np.vectorize(f)(pred_df.index.values,
                            pred_df[var].values)

    # a function to save a xarray dataset into disk
    def _to_file(self,
                 dataset: xr.Dataset,
                 output_dir: str | Path,
                 mode: str | None = None):

        # only supports zarr and netcdf
        suffix = output_dir.suffix.lower()
        if suffix == '.zarr':
            m = mode if mode else 'r+'
            dataset.to_zarr(output_dir,
                            mode=m,
                            **self.kwargs)

        elif suffix == '.nc':
            m = mode if mode else 'a'
            dataset.to_netcdf(output_dir,
                              mode=m,
                              **self.kwargs)

        else:
            message = f"Unknown file extension {suffix}. "
            message += "The file writer only supports zarr "
            message += "and netcdf extensions."
            raise NotImplementedError(message)

    # a function to read a xarray dataset from disk
    def _open_file(self,
                   output_dir: str | Path,
                   **kwargs):

        # only supports zarr and netcdf
        suffix = output_dir.suffix.lower()
        if suffix == '.zarr':
            return xr.open_zarr(output_dir, **kwargs)

        elif suffix == '.nc':
            return xr.open_dataset(output_dir, **kwargs)

        else:
            message = f"Unknown file extension {suffix}. "
            message += "The file reader only supports zarr "
            message += "and netcdf extensions."
            raise NotImplementedError(message)

    def archive(self,
                predictions: dict[str, np.ndarray]):
        """Insert the predicted values into the corresponding coordinates
           and save the result into disk

        Parameters
        ----------
        predictions         : dict[str, np.ndarray]

        Dictionary of the model predictions that contains predicted variables
        and their corresponding coordinates.

        Returns
        -------

        Raises
        ------
        ValueError
            If the specified coordinate keys in the predictions
            dictionary does not match the coordinates of the dataset
            that is used to store values into. For example, if the
            keys of the predictions dictionary are
            ['variable1', 'latitude', 'longitude', 'time'] and the
            coordinates of the xarray dataset are
            ['latitude', 'longitude'].

        """

        # read the target xarray dataset from disk. This line
        # is like a file refresh to get the latest updates of
        # the file.
        output_data = self._open_file(self.output_path)

        if not set(self.coords).issubset(set(list(predictions.keys()))):
            message = 'All coordinates of the data_schema must '
            message += 'be included in the predictions.'
            raise ValueError(message)

        # reading netcdf changes the order of coords, this line reorders
        # the coords to the original one
        output_data = output_data[self.coords + list(output_data.keys())]

        # make sure that all the predicted variables exist in the output
        # dataset
        for var in [k for k in predictions.keys() if k not in self.coords]:
            if var not in list(output_data.keys()):
                output_data[var] = (list(output_data.coords),
                                    da.full(shape=tuple(output_data.sizes.values()),
                                            fill_value=np.nan)
                                    )
        self._to_file(dataset=output_data,
                      output_dir=self.output_path,
                      mode='a')

        # This line is just removing the extra dim from the values
        # of the predictions dict
        for k in predictions.keys():
            if len(predictions[k].shape) > 1:
                predictions[k] = np.squeeze(predictions[k])

        # coordinate matching
        mc = self._match_coords(pred_data=predictions)

        # aggregate the predicted values of a same coordinate
        df = self._aggregate(pred_data=mc)

        # insert the result into the output dataset
        self._insert_values(pred_df=df,
                            dataset=output_data)

        # save the updated dataset into disk
        self._to_file(dataset=output_data,
                      output_dir=self.output_path)
