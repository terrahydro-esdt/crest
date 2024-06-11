import logging
import shutil
import typing
import ast
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
import dask.array as da
import dask

from crest.base.BaseAbstract import BaseAbstract
from crest.data.loading.Dataset import Dataset
from crest.data.loading.Datafile import Datafile

# to supress a warning about the large chunk indexing
dask.config.set(**{'array.slicing.split_large_chunks': True})

logger = logging.getLogger(__name__)  # create logger here or...


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
        
    coords              : list
        The list of the coordinates of the predicted values. e.g.
        for a spatiotemporal predictions it is 
        ['datetime', 'latitude', 'longitude']

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
    
    overwrite           : bool, Optional
        If the output_path exists, whether the file must be overwritten.
        Default is False.

    **kwargs
        Any additional kwargs are passed into xr.Dataset.to_netcdf
        if the final output must be saved as a netcdf file.

    Raises
    ------
    FileExistsError
        - When temp.zarr file exists in the output_path directory.
          This file name is saved for the Archiver process when
          the requested output file format is something other than
          zarr.

    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __init__(self,
                 output_path: str | Path,
                 coords: list[str],
                 data_schema: Datafile | Dataset | dict[str, np.ndarray] | None = None,
                 datafile_index: int | None = None,
                 aggregate_func: typing.Callable = lambda x: np.mean(
                     x, axis=0),
                 exact_coord_match: bool = False,
                 overwrite: bool = False,
                 **kwargs
                 ):

        self.output_path = output_path
        self.coords = coords
        self.data_schema = data_schema
        self.datafile_index = datafile_index
        self.aggregate_func = aggregate_func
        self.exact_coord_match = exact_coord_match
        self.overwrite = overwrite
        self.kwargs = kwargs

        self.pred_val = False

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # set/convert the suffix of the output file to zarr
        # we only archive to zarr format and create other
        # formats like netcdf after archiving if asked
        self.output_suffix = self.output_path.suffix.lower()
        if self.output_suffix != '.zarr':
            par = self.output_path.parent
            self.output_path_zarr = par / 'temp.zarr'

            if self.output_path_zarr.exists():
                message = 'When the requested file format is not '
                message += 'zarr, the archiver creates a temporary '
                message += 'file of temp.zarr for the process. '
                message += f'temp.zarr already exists at {par}. '
                raise FileExistsError(message)
        else:
            self.output_path_zarr = self.output_path

        self.out_datafile = self._create_schema()

        # create the KDTree if the exact coord matching is asked
        if self.exact_coord_match:
            self.lookup_space = self._create_coord_search_space()
            self.s = KDTree(self.lookup_space)

    def __exit__(self, *args, **kwargs):
        self.close(origin='__exit__')

    def __enter__(self):
        return self

    # a function to create the data_schema
    def _create_schema(self):

        if self.data_schema:
            if isinstance(self.data_schema, dict):
                out_datafile = xr.Dataset(coords=self.data_schema)

            else:
                # if the data_schema is of type Dataset
                if isinstance(self.data_schema, Dataset):

                    # By default, the finest resolutions among all datafiles
                    # are used to make the data_schema unless a specific datafile
                    # is specified by the user.
                    if not self.datafile_index:
                        all_coords = list(set(c for d in self.data_schema for c in d.coords if c != 'features'))
                        if not set(self.coords).issubset(set(all_coords)):
                            message = f'All items of the coords {set(self.coords)} must exist '
                            message += f'in the data_schema {set(all_coords)}.'
                            raise ValueError(message)

                        finest_resolution = dict.fromkeys(self.coords, np.inf)
                        finest_min = dict.fromkeys(self.coords, np.NINF)
                        finest_max = dict.fromkeys(self.coords, np.inf)
                        finest_coordinate = dict.fromkeys(self.coords)
                        for d in self.data_schema:
                            res_val = [np.mean(res) if isinstance(res, np.ndarray) else res for res in d.resolution]
                            current_res = dict(zip(list(d.coords), res_val))

                            for cor, val in current_res.items():
                                if val != 0.:
                                    coord_array = d.coords[cor].to_numpy().astype(np.float64)
                                    if val < finest_resolution[cor]:
                                        finest_resolution[cor] = val
                                        finest_coordinate[cor] = coord_array

                                    if np.max(coord_array) <= finest_max[cor]:
                                        finest_max[cor] = np.max(coord_array)

                                    if np.min(coord_array) >= finest_min[cor]:
                                        finest_min[cor] = np.min(coord_array)

                        # find the finest resolution and here, limit the coordinates of the finest
                        # resolution to the smallest range among the datafiles        
                        finest_coordinate = {k: val[val <= finest_max[k]] for k, val in finest_coordinate.items()}
                        finest_coordinate = {k: val[val >= finest_min[k]] for k, val in finest_coordinate.items()}
                        self.logger.info(finest_coordinate)
                        out_datafile = xr.Dataset(coords=finest_coordinate)

                    # if the specific datafile must be data schema, 
                    # it is captured from the dataset
                    else:
                        if self.datafile_index < 0 or self.datafile_index >= len(self.data_schema):
                            message = 'datafile_index is out of range. '
                            message += 'The valid range is '
                            message += f'(0, {len(self.data_schema) - 1})'
                            raise ValueError(message)

                        out_datafile = self.data_schema[self.datafile_index]

                # if the data schema is a datafile      
                else:
                    out_datafile = self.data_schema

                # if the data schema is of type datafile,
                # convert it to an empty xarray dataset
                if isinstance(out_datafile, Datafile):
                    # convert the features coordinate to separate variables
                    out_datafile = out_datafile.data.to_dataset(dim='features')
                    # create the empty output dataset using the schema and save it to
                    # the output_dir
                    out_datafile = xr.full_like(out_datafile,
                                                np.nan).drop_vars(list(out_datafile.keys()))

            # check if a file with output_path already exists
            if self.output_path.exists():

                # if the user asks for overwriting the existing file
                if self.overwrite:
                    shutil.rmtree(self.output_path)

                # if overwriting is not asked and the specified data schema
                # has the exact same structure as the file in output_path,
                # the file is loaded and serves as the data_schema.
                # If the structures conflict, an error is raised
                else:
                    existing_file = self._open_file(output_dir=self.output_path)
                    existing_file_empty = xr.full_like(existing_file, np.nan).drop_vars(list(existing_file.keys()))
                    if out_datafile.equals(existing_file_empty):

                        message = 'The data_schema matches the structure of the '
                        message += 'file already exists at the output_path. '
                        message += 'The existing file serves as the data_schema.'
                        logging.warning(message)
                        out_datafile = existing_file

                    else:
                        message = f'{self.output_path}  with a different'
                        message += ' structure than the specified data_schema '
                        message += 'already exists. For overwriting, use overwrite=True.'
                        raise FileExistsError(message)

            else:
                message = f'initializing {self.output_path_zarr.name} at '
                message += f'{str(self.output_path_zarr.parent)} ...'
                self.logger.info(message)
                self._to_file(out_datafile,
                              self.output_path_zarr,
                              mode='w')
                self.logger.info('Done')

        # the case that the data_schema is not specified,
        # check for any existing file at output_path to
        # use as data_schema.
        else:
            if self.output_path.exists():
                message = f'loading {self.output_path.name} from '
                message += f'{str(self.output_path.parent)}'
                self.logger.info(message)
                out_datafile = self._open_file(output_dir=self.output_path)
                self.logger.info('Done')

            else:
                message = 'data_schema must be specified if '
                message += 'output_path does not exist'
                raise ValueError(message)

        out_datafile.attrs = {}
        if set(list(out_datafile.coords)) != set(self.coords):
            message = 'The coordinates of the data_schema must '
            message += 'be identical to the coords. '
            message += f'{set(list(out_datafile.coords))} '
            message += f'versus {set(self.coords)}'
            raise ValueError(message)

        return out_datafile

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
                   pred_df: pd.DataFrame
                   ) -> pd.DataFrame:

        # group the same rows and aggregate them by the specified
        # function
        variables = [col for col in pred_df.columns if col not in self.coords]
        pred_df = pred_df.groupby(self.coords, sort=False)
        pred_df = pred_df.apply(
            func=lambda x: self.aggregate_func(x))[variables]

        return pred_df

    def _insert_values(self,
                       grouped_df: pd.DataFrame):
        """Insert the predicted values into the corresponding coordinates
           and save the result into disk.
           
           This functions shows the best performance when it gets
           the prediction batches grouped over coordinates so it
           write them over a region of the output zarr file. For instance,
           if the predictions are in a Three dimensional space, the groups should
           be like:
           coord1_value_1    coord2_value_1    coord3_value_1
                                               coord3_value_2
                                               coord3_value_3
                                               coord3_value_4
                                               ...
                                               coord3_value_n
                                               
          coord1_value_1    coord2_value_2     coord3_value_1
                                               coord3_value_2
                                               coord3_value_3
                                               coord3_value_4
                                               ...
                                               coord3_value_n
        Parameters
        ----------
        grouped_df         : pd.DataFrame

        A pandas DataFrame with index as the coodinates of the model predictions
        and columns as the model predictions. if the data has multiple coordinates,
        grouped_df would be multiIndex.

        Returns
        -------

        Raises
        ------

        """

        # get all coordinates except for the last one
        # and get their groups
        levels = list(range(len(self.coords) - 1))
        groups = grouped_df.groupby(level=levels)
        group_keys = list(groups.groups.keys())
        index_name = grouped_df.index.names

        # create an empty dict so later specify the range of the
        # xarray dataset that must be written by the predictions
        slices = {}

        # loop over the groups of coordinates
        for l_group in group_keys:

            # get the rows with selected group
            grouped_df_sel = groups.get_group(l_group)

            # find the index of the selected coordinate value within the
            # output xarray dataset. for all of the coordinate except
            # the last one, we only have one value so we retrieve only one
            # index.
            for index, dim in enumerate(l_group):
                l_index = int(np.where(self.out_datafile[index_name[index]] == dim)[0])
                slices[index_name[index]] = slice(l_index, l_index + 1)

            # for the last coordinate we have a range of values so we get a range
            # of coordinates
            lev = grouped_df_sel.reset_index()[index_name[len(self.coords) - 1]]
            ind_min = int(
                np.where(self.out_datafile[lev.name] == min(lev))[0])
            ind_max = int(
                np.where(self.out_datafile[lev.name] == max(lev))[0])
            slices[lev.name] = slice(ind_min, ind_max + 1)

            # capture the region of the output xarray dataset by the indices
            # of the coodinates and insert the model predictions into them
            data_schema_sel = self.out_datafile.isel(
                slices).to_dataframe().loc[:, []]
            data_schema_sel = data_schema_sel.join(
                grouped_df_sel, how='left').to_xarray()

            # try to save the updated region into disk
            try:
                self._to_file(data_schema_sel,
                              self.output_path_zarr,
                              region=slices,
                              mode='r+')

            # handle the errors regarding inconsistent staructure
            # of the output dataset and updated region
            except ValueError as e:
                e = str(e)

                # check if the coordinates are consistent
                message = r'Dimensions[^"]*do not exist. Expected one or '
                message += r'more of Frozen[^"]*'
                if re.search(message, e):
                    message = 'All coordinates of the data_schema must '
                    message += 'be included in the predictions.'
                    raise ValueError(message)

                # check if the predicted variables exist in the output dataset
                message = r"dataset contains non-pre-existing variables \[[^\]]+\], "
                message += "which is not allowed in ``xarray\.Dataset\.to_zarr\(\)`` with "
                message += "mode='[^']+?'.*?To allow writing new variables, set mode='[^']+?'."
                if re.search(message, e):

                    # if the predicted variables do not exist in the output dataset
                    # create them
                    vars_to_add = ast.literal_eval(
                        e.split('variables ')[1].split(', which')[0])
                    for c in vars_to_add:
                        self.out_datafile[c] = (self.coords,
                                                da.full(shape=tuple(self.out_datafile.sizes.values()),
                                                        fill_value=np.nan)
                                                )

                    self.out_datafile = self.out_datafile.chunk('auto')
                    self._to_file(self.out_datafile,
                                  self.output_path_zarr,
                                  mode='a')

                    # try saving the region to disk again
                    self._to_file(data_schema_sel,
                                  self.output_path_zarr,
                                  region=slices,
                                  mode='r+')

    # a function to convert the zarr file to other formats
    def _to_file(self,
                 dataset: xr.Dataset,
                 output_dir: str | Path,
                 mode: str | None = None,
                 region: dict | None = None):

        # only supports netcdf and zarr
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        suffix = output_dir.suffix.lower()

        if suffix == '.zarr':
            dataset.to_zarr(output_dir,
                            mode=mode,
                            region=region,
                            **self.kwargs)

        if suffix == '.nc':
            mode = mode if mode else 'w'
            dataset.to_netcdf(output_dir,
                              mode=mode,
                              **self.kwargs)

        else:
            message = f"Unknown file extension {suffix}. "
            message += "The file writer only supports netcdf"

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
        """concatenate the predictions of a unique coordinate
           into a single dictionary and archive it.

           This function is necessary because in some cases there are
           more than one predictions for a specific coordinate
           and the archiver aggregate them into a single
           value before the "insert into the corresponding coordinate"
           step. So, all of the predictions of a unique coordinate
           must be in a single archiver input.

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

        # check of the coordinates of the predictions matches the
        # specified data_schema
        if not set(self.coords).issubset(set(list(predictions.keys()))):
            message = 'All coordinates of the data_schema must '
            message += 'be included in the predictions.'
            raise ValueError(message)

        # remove any unwanted axis from the prediction dict
        p_dt_dict = {k: np.squeeze(i) for k, i in predictions.items()}

        # match the coordinates of the predictions with the
        # output dataset
        p_dt_dict = self._match_coords(p_dt_dict)

        # handle the cases that only one row exists
        if np.sum([len(v.shape) for v in p_dt_dict.values()]) == 0.:
            p_dt_dict = {k: v[np.newaxis] for k, v in p_dt_dict.items()}

        # specify the pred_val attribute by the new predictions
        # if the pred_val is empty.
        if isinstance(self.pred_val, bool):
            self.pred_val = pd.DataFrame.from_dict(p_dt_dict)

        # If pred_val is not empty, compare the coordinates
        # of the last row in pred_val with the coordinates of
        # the first row in the new batch of predictions.
        # All coordinates except the last one are relevant here.
        # This distinction is important because writing to the
        # zarr file involves a region where the first (n-1)
        # coordinates use a single index, while the coordinate
        # n uses a range of indices.
        else:
            # handle the cases where we only have one dimension
            coords_subset = self.coords[:-1] if len(self.coords) > 1 else self.coords

            last_row = self.pred_val.iloc[-1, :][coords_subset]
            first_row = pd.DataFrame.from_dict(
                p_dt_dict).iloc[0, :][coords_subset]

            # if the rows has the same coodinates, we extract the last
            # same n rows of the pred_val and prepend it to the
            # new prediction batch. We keep this for future use and 
            # archive the rest of the rows of pred_val.
            if np.sum(last_row == first_row) == len(coords_subset):
                extension_check = self.pred_val[coords_subset] == last_row
                extension_check = extension_check.sum(axis=1)

                p_dt = self.pred_val[extension_check != len(
                    coords_subset)].reset_index(drop=True)
                self.pred_val = pd.concat([self.pred_val[extension_check == len(coords_subset)],
                                           pd.DataFrame.from_dict(p_dt_dict)]).reset_index(drop=True)

            # otherwise vacate pred_val and archive the new prediction batch
            else:
                p_dt = self.pred_val.copy()
                self.pred_val = False

            # In some cases, the whole prediction batch is prepended
            # to the next batch so p_dt will be empty.
            if len(p_dt.index) != 0:
                # before archiving, aggregate the rows to handle
                # multiple predictions for a single coordinate
                g_dt = self._aggregate(p_dt)
                # and then archive
                self._insert_values(g_dt)

    def close(self, origin: str = ''):
        """Archive the last batch of predictions and handle
           the final output file extension. if the requested
           output file is something other than zarr (e.g. nc),
           convert the archived zarr file to the requested
           extension and delete it.
        """

        if not isinstance(self.pred_val, bool):
            g_dt = self._aggregate(self.pred_val)
            self._insert_values(g_dt)

        # if the requested output file is something other
        # than zarr like netcdf(.nc), convert the archived
        # zarr file to the requested extension and delete it
        if self.output_suffix != '.zarr':
            self._to_file(self._open_file(self.output_path_zarr),
                          self.output_path,
                          mode=None
                          )

            shutil.rmtree(self.output_path_zarr)
