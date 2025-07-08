import logging
import shutil
from pathlib import Path
import numpy as np
import xarray as xr
import dask

import s3fs
import boto3
import json
from botocore.exceptions import NoCredentialsError
from crest.utils.S3Path import S3Path

from crest.base.BaseAbstract import BaseAbstract
from crest.data.loading import Dataset
from crest.data.loading import Datafile

# to supress a warning about the large chunk indexing
dask.config.set(**{'array.slicing.split_large_chunks': True})

logger = logging.getLogger(__name__)


# a function to get an index of a datafile in a dataset
# using the name of its zarr file source
def _get_index_from_name(df_name: str, dataset):
    if dataset is None:
        message = 'To capture the coordinate by the names'
        message += ' of the datafiles, specify the crest dataset'
        raise ValueError(message)
    
    try:
        return next(i for i, df in enumerate(dataset) if df_name in df.name.lower())
    except StopIteration:
        raise ValueError(f'{df_name} not found')

class Archiver(BaseAbstract):
    """Class which handles inserting the model predictions into the
       corresponding coordinates of an xarray dataset and write the
       result into disk.
       
       In some cases, multiple predictions exist for the same time and
       location due to differences in the grid structures of the CREST
       data files. To handle this, the archiver computes summary statistics
       (e.g., sum and count for averaging) across those predictions.
       It then generates two output files:
       
       - A statistics file in Zarr format, which contains the necessary
       statistical values used to compute the final result.
       
       - The final archived output file, which is produced by applying
       those statistics and saved in either Zarr or NetCDF format.

    Parameters
    ----------
    output_path         : str | Path,

        This argument specifies the path to the output file, which can be
        in either zarr ('.zarr') or netcdf ('.nc') formats. The class
        will automatically determine the file format based on the provided
        path. If no file extension is given, it will default to using the
        zarr format. The dataset located at the output_path (if exists) is
        treated as the default data_schema for storing predicted
        values and it must have at least one variable. output_path
        can also be an s3 URI starts with 's3://'.

    data_schema      : Datafile | Dataset | dict, Optional
        This argument pertains to a Datafile used for storing predicted
        values. The user has the options of a single Datafile,
        a Dataset, which is essentially a list of Datafiles or a dictionary
        with the keys being the coordinate names and the values being the
        arrays of coordinate values.If a Dataset is used, the user could
        also indicate which specific datafile to use by specifying the
        datafile_index, or else for each coordinate the finest resolution among
        the datafiles is used. The predicted values are then inserted into
        the nearest coordinate within the chosen Datafile. If the output_path
        exists, the specified data_schema is ignored.

    datafile_index      : int | str | dict[str, int | str], Optional
        The index of the Datafile to use as the data_schema when a Dataset
        is provided. datafile_index can be either a dictionary, an integer or a string
        representing the name of the datafile. The datafile name is a two-part
        string separated by an underscore (_). The user may specify either part,
        but typically the first part is human-readable and derived from the
        name of the source data's Zarr file. datafile_index could also be a
        dictionary with the keys being the name of the coordinates and the
        values being either the name of the datafile or its index as integer.
    
    overwrite           : bool
        If the output_path exists, whether the file must be overwritten.
        Default is False.
        
    per_sample          : bool
        per_sample=True runs archiving for each sample of a batch separately.
        This option could be helpful for the case that the samples are large ndarrays
        and using per_sample reduces some extra computations by directly insert the
        sample into the output file. Default is False.
    
    task_bytes          : float, Optional
        A float that shows how much memory should be allocated to the
        archiver. The archiver keeps all the computations in memory until
        reaching the task_bytes limits and then it insert them into the
        output file.
    
    keep_stats          : bool
        Whether the zarr file that contains the statistics should be
        kept after that the archiver finishes. Default is True.
        For the future update of an output file, the keep_stats
        must be True and the stats zarr file must exist or a FileNotFound
        error will raise.

    aws_credentials_path : str, optional
        If the output_path is an S3 URI (starts with s3://), the path to 
        an aws_credentials txt file is required unless a 'cred_cache.json'
        file with the valid 'key', 'secret', 'token' values exists.
    
    **kwargs
        Any additional kwargs are passed into xr.Dataset.to_netcdf
        if the final output must be saved as a netcdf file.

    Raises
    ------
    ValueError
        - When the file extension of the output_path is not nc or zarr.
        - When the output_path is an s3 URI but aws_credentials_path is None.
    NotImplementedError
        - When the output_path is an s3 URI but the file extension is nc.
    FileNotFoundError
        - When aws_credentials_path does not exist.
    FileExistsError
        - When the output_path directory does not exist.

    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __init__(self,
                 output_path: str | Path,
                 data_schema: Datafile | Dataset | dict | None = None,
                 datafile_index: int | str | dict[str, int | str] | None = None,
                #  aggregate_func: str = 'mean',
                 overwrite: bool = False,
                 per_sample: bool = False,
                 task_bytes: float | None = None,
                 keep_stats: bool = True,
                 aws_credentials_path: str | None = None,
                 **kwargs
                 ):

        self.output_path = output_path
        self.data_schema = data_schema
        self.datafile_index = datafile_index
        # self.aggregate_func = aggregate_func
        self.overwrite = overwrite
        self.per_sample = per_sample
        self.task_bytes = task_bytes
        self.keep_stats = keep_stats
        self.aws_credentials_path = aws_credentials_path
        self.kwargs = kwargs
        
        # initialize an empty var to keep the computations
        # in memory for the task_bytes.
        self.data_in_memory = None
        
         # handle the output_path for being either s3 or local
        if isinstance(self.output_path, str):
            suffix = self.output_path.split('.')[-1].lower()
            if suffix not in ['nc', 'zarr']:
                raise ValueError(f'invalid output file extension: {suffix}')
            if self.output_path.startswith('s3://'):
                # cannot directly ingest into netcdf in s3
                if suffix == 'nc':
                    message = 'Saving to NetCDF format is not supported when '
                    message+= 'the output_path is an S3 bucket'
                    raise NotImplementedError(message)
                try:
                    S3Path(self.output_path).exists()
                    
                # when the credential is needed for s3 (e.g.
                # accessing bucket from a local system) either
                # cached credentials in json file or
                # credentials in txt file is needed. When
                # a bucket is accessed from an aws instance,
                # no credentials are needed.
                except NoCredentialsError:
                    if Path('./cred_cache.json').exists():
                        with open("./cred_cache.json", "r") as f:
                            aws_credentials = json.load(f)

                    else:
                        if self.aws_credentials_path is None:
                            raise ValueError('AWS credentials are needed but aws_credentials_path is None')

                        # prepare the credentials secret keys from a txt file if needed
                        if not Path(aws_credentials_path).exists():
                            message = f'AWS credentials file missing at {aws_credentials_path}'
                            raise FileNotFoundError(message)
                            
                        key, secret, mfa_serial = Path(aws_credentials_path).read_text().strip().split(',')
                        mfa_kwargs = {
                            'DurationSeconds': 129600,
                            'SerialNumber': mfa_serial,
                            'TokenCode': input('Enter AWS MFA code: '),
                        }
                        session = boto3.Session(key, secret)
                        mfa_auth = session.client('sts').get_session_token(**mfa_kwargs)
                        aws_credentials =  {
                            'key': mfa_auth['Credentials']['AccessKeyId'],
                            'secret': mfa_auth['Credentials']['SecretAccessKey'],
                            'token': mfa_auth['Credentials']['SessionToken'],
                        }
                        
                        # make the credentials cache file
                        with open("./cred_cache.json", "w") as f:
                            json.dump(aws_credentials, f, indent=4) 
                    
                    # make the s3 mapper for the output directory
                    store  = s3fs.S3FileSystem(**aws_credentials)
                    self.output_path = store.get_mapper(self.output_path)
                
                self.output_path = S3Path(self.output_path)
            
            else:
                self.output_path = Path(self.output_path)

        else:
            suffix = self.output_path.name.split('.')[-1].lower()
            if suffix not in ['nc', 'zarr']:
                raise ValueError(f'invalid output file extension: {suffix}')
            
        
        if not self.output_path.parent.exists():
            raise FileExistsError(f'{self.output_path.parent} does not exist')
        # specify the name of the stats zarr file
        self.output_path_stats = self.output_path.parent.joinpath(f'{self.output_path.stem}_stats.zarr')

        # create the data schema
        self.out_datafile = self._create_schema()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
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
                        coords = list(set(c for d in self.data_schema for c in d.coords if c != 'features'))

                        finest_resolution = dict.fromkeys(coords, np.inf)
                        finest_min = dict.fromkeys(coords, np.NINF)
                        finest_max = dict.fromkeys(coords, np.inf)
                        finest_coordinate = dict.fromkeys(coords)
                        finest_df_name = dict.fromkeys(coords)
                        dt_types_coords = []
                        for d in self.data_schema:
                            for c_check in list(d.coords):
                                if np.issubdtype(d.data[c_check].dtype, np.datetime64) and c_check not in dt_types_coords:
                                    dt_types_coords.append(c_check)
                            
                            res_val = [np.mean(res) if isinstance(res, np.ndarray) else res for res in d.resolution]
                            current_res = dict(zip(list(d.coords), res_val))

                            for cor, val in current_res.items():
                                if val != 0.:
                                    coord_array = d.coords[cor].to_numpy().astype(np.float64)
                                    if val < finest_resolution[cor]:
                                        finest_resolution[cor] = val
                                        finest_coordinate[cor] = coord_array
                                        finest_df_name[cor] = d.name.split('_')[0]

                                    if np.max(coord_array) <= finest_max[cor]:
                                        finest_max[cor] = np.max(coord_array)

                                    if np.min(coord_array) >= finest_min[cor]:
                                        finest_min[cor] = np.min(coord_array)

                        # find the finest resolution and here, limit the coordinates of the finest
                        # resolution to the smallest range among the datafiles
                        finest_coordinate = {k: val[val <= finest_max[k]] for k, val in finest_coordinate.items()}
                        finest_coordinate = {k: val[val >= finest_min[k]] for k, val in finest_coordinate.items()}

                        for k, val in finest_coordinate.items():
                            if k in dt_types_coords:
                                finest_coordinate[k] = val.astype('datetime64[ns]')

                        self.logger.info(f'The following datafiles are used for making the data schema: {finest_df_name}')
                        
                        out_datafile = xr.Dataset(coords=finest_coordinate)

                    # if the specific datafile must be data schema,
                    # it is captured from the dataset
                    else:
                        if isinstance(self.datafile_index, dict):
                            new_schema = {}
                            for k, v in self.datafile_index.items():
                                index_of_df = _get_index_from_name(v, self.data_schema) if isinstance(v, str) else v
                                new_schema[k] = self.data_schema[index_of_df].data.coords[k]
                            out_datafile = xr.Dataset(coords=new_schema)
                        else:     
                            if isinstance(self.datafile_index, str):
                                self.datafile_index = _get_index_from_name(self.datafile_index,
                                                                        self.data_schema)
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

            
            out_datafile = out_datafile.expand_dims(stats=['sum', 'count'])
            # check if a file with output_path already exists
            if self.output_path.exists():

                # if the user asks for overwriting the existing file
                if self.overwrite:
                    self._delete_existing(self.output_path)
                    if self.output_path_stats.exists():
                        self._delete_existing(self.output_path_stats)
                    

                # if overwriting is not asked and the specified data schema
                # has the exact same structure as the file in output_path,
                # the file is loaded and serves as the data_schema.
                # If the structures conflict, an error is raised
                else:
                    if not self.output_path_stats.exists():
                        message  = f'{self.output_path_stats} not found. '
                        message += 'To update the output file with new predictions, '
                        message += 'the stats file must be provided. Provide the stats '
                        message += 'file, use a different output name or set overwrite=True.'
                        raise FileNotFoundError(message)
                    else:
                        existing_file = self._open_file(output_dir=self.output_path_stats)
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
                        
                    if self.output_path.exists():
                        self._delete_existing(self.output_path)

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
        
        return out_datafile

    # a function to convert the zarr file to other formats
    def _to_file(self,
                 dataset: xr.Dataset,
                 output_dir: str | Path | S3Path,
                 mode: str | None = None,
                 region: dict | None = None):

        # only supports netcdf and zarr
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        suffix = output_dir.name.split('.')[-1].lower()

        if suffix == 'zarr':
            dataset.to_zarr(output_dir,
                            mode=mode,
                            region=region,
                            **self.kwargs)

        if suffix == 'nc':
            mode = mode if mode else 'w'
            dataset.to_netcdf(output_dir,
                              mode=mode,
                              **self.kwargs)

    # a function to remove either local or s3 zarr files
    def _delete_existing(self,
                         file_path: Path | S3Path):
        if isinstance(file_path, S3Path):
            file_path.delete()
        else:
            if file_path.name.split('.')[-1] == 'nc':
                file_path.unlink()
            else:
                shutil.rmtree(file_path)
            
    # a function to read a xarray dataset from disk
    def _open_file(self,
                   output_dir: str | Path | S3Path,
                   **kwargs):

        # only supports zarr and netcdf
        suffix = output_dir.name.split('.')[-1].lower()
        if suffix == 'zarr':
            return xr.open_zarr(output_dir, **kwargs)

        elif suffix == 'nc':
            return xr.open_dataset(output_dir, **kwargs)
    
    # a function to add axis to pred dict if the number
    # of axis in pred is not equal to the number of coords
    def _match_shape(self,
                     coord_dict: dict[str, np.ndarray],
                     pred_dict : dict[str, np.ndarray],):
        
        for k, v in pred_dict.items():
            missing = len(coord_dict) - (v.ndim-1)
            if missing > 0:
                pred_dict[k] = np.reshape(v, v.shape + (1,) * missing)
        
        return pred_dict
    
    # a function to flattens all the samples in a batch
    # so we have a prediction dict in which each sample
    # is for a specific nd pixel.
    def _flatten(self,
                 coord_dict: dict[str, np.ndarray],
                 pred_dict : dict[str, np.ndarray],):

        pred_dict = self._match_shape(coord_dict, pred_dict)
        batch = coord_dict | pred_dict
        
        # take the batch_size out of the shape tuple
        # to know how to reshape the arrays (coords and preds)
        ref_shape = batch[list(pred_dict.keys())[0]].shape
        n = ref_shape[0]
        ref_shape = batch[list(pred_dict.keys())[0]].shape
        spatial_shape = ref_shape[1:]
        flat_size = np.prod(spatial_shape)
        
        # reshape the preds
        flat_vars = {
        k: (['sample'], batch[k].reshape(int(n * flat_size)))
        for k in pred_dict.keys()
        }
        
        # reshape the coords
        selected_axis = []
        flat_coords = {}
        for k in list(coord_dict.keys()):
            coord = batch[k]
            
            # if all axes of the coords are of size 1 use the same thing
            if coord.ndim == 1:
                flat_coords[k] = (['sample'], coord)
                continue
            
            coord_dim = batch[k].shape[-1]

            axis_idx = [i for i, s in enumerate(ref_shape[1:]) if s == coord_dim]
            axis_idx = [a for a in axis_idx if a not in selected_axis]
            # we also need to specify which axis we already used
            # because in some cases two axis have the same size
            selected_axis.append(axis_idx[0])

            if not axis_idx:
                raise ValueError(f"Could not map coordinate '{k}' to any axis of variable shape {ref_shape}")

            shape = [1] * len(ref_shape)
            shape[0] = batch[k].shape[0]
            shape[axis_idx[0] + 1] = coord_dim

            broadcasted = np.broadcast_to(coord.reshape(shape), ref_shape)
            flat_coords[k] = (['sample'], broadcasted.reshape(-1))
            
        return flat_coords, flat_vars
    
    # a function to get the sum and count
    # (TODO: stats in general)
    def _flatten_sum_count(self,
                           coord_dict_flat: dict,
                           pred_dict_flat : dict,):
        
        # create a non-uniform grid from the pred and coord dict
        ds = xr.Dataset(data_vars=pred_dict_flat, coords=coord_dict_flat)

        # get the stats
        ds = ds.set_index(sample=list(coord_dict_flat.keys()))
        ds_sum = ds.groupby('sample').sum(dim=...).unstack()
        ds_count = ds.groupby('sample').count(dim=...).unstack()
        
        # if the coords are different than the schema coords values
        # then use nearest method to insert them values in correct
        # coordinate
        method = None if all(ds_sum.coords[dim].isin(self.out_datafile.coords[dim]).all() for dim in ds_sum.coords) else 'nearest'

        # reshape the non-uniform grid to the schema shape
        ds_sum = ds_sum.reindex_like(self.out_datafile, method=method)
        ds_count = ds_count.reindex_like(self.out_datafile, method=method)
        
        sum_array = xr.concat([ds_sum[k] for k in pred_dict_flat.keys()], dim='features')
        count_array = xr.concat([ds_count[k] for k in pred_dict_flat.keys()], dim='features')

        sum_array = sum_array.assign_coords(features=list(pred_dict_flat.keys()))
        count_array = count_array.assign_coords(features=list(pred_dict_flat.keys()))

        final = xr.concat([sum_array, count_array], dim='stats')
        final = final.assign_coords(stats=['sum', 'count'])
        
        return final
    
    # a version of getting the sum and count
    # function specific for the per-sample=True
    # case. in this setup we don't flatten anything.
    def _sample_sum_count(self,
                          coord_dict: dict[str, np.ndarray],
                          pred_dict : dict[str, np.ndarray],
                          sample_index: int,):
        
        pred_dict = self._match_shape(coord_dict, pred_dict)

        batch = coord_dict | pred_dict
        stat_arrays = []
        for k in pred_dict.keys():
            data = xr.DataArray(
                batch[k][sample_index],
                dims=list(coord_dict.keys()),
                coords={dim: batch[dim][sample_index] for dim in coord_dict.keys()}
            )
            ds_sum = data
            ds_count = (~data.isnull()).astype("int64")
                
            method = None if all(ds_sum.coords[dim].isin(self.out_datafile.coords[dim]).all() for dim in ds_sum.coords) else 'nearest'

            ds_sum = ds_sum.reindex_like(self.out_datafile, method=method)
            ds_count = ds_count.reindex_like(self.out_datafile, method=method)
            
            stacked = xr.concat([ds_sum, ds_count], dim="stats")
            stacked = stacked.assign_coords(stats=["sum", "count"])
            stat_arrays.append(stacked)

        final = xr.concat(stat_arrays, dim="features")
        final = final.assign_coords(features=list(pred_dict.keys()))
        
        return final
    
    # a function to update the sum and count
    # after each batch. The problem was
    # if we use normal summation function
    # with the xarray datasets it does includes
    # the NaNs
    def _update_data(self,
                     data1: xr.DataArray,
                     data2: xr.DataArray,):
        
        data_sum = xr.where(data1.notnull(), data1, 0) + xr.where(data2.notnull(), data2, 0)
        data_sum = data_sum.where(data1.notnull() | data2.notnull())
        
        return data_sum
    
    # a function to insert the computed sum and count
    # into the stats zarr file
    def _insert_output(self,
                       data: xr.DataArray,):
        
        # if we don't have task_bytes then we push the
        # changes to the stats zarr file immediately
        if self.task_bytes is None:
            if self.output_path_stats.exists():
                data1 = self._open_file(self.output_path_stats)
                data1 = data1.to_array(dim='features')
                if set(data1.to_dataset('features').data_vars.keys()) != set(data.to_dataset('features').data_vars.keys()):
                    message = 'All coordinates of the data_schema must '
                    message += 'be included in the coord_dict.'
                    raise ValueError(message)
                data = self._update_data(data1, data)
            
            self._to_file(data.to_dataset('features'),
                          self.output_path_stats,
                          mode='a')
        
        # if we have task_bytes then we wait until
        # we reach the memory limit and we don't
        # update the zarr file until then
        else:
            if self.data_in_memory is None:
                self.data_in_memory = data
                if self.data_in_memory.nbytes / (1024 ** 3) >= self.task_bytes:
                    self._to_file(self.data_in_memory.to_dataset('features'),
                                  self.output_path_stats,
                                  mode='a')
                    self.data_in_memory = None

            else:
                if self.data_in_memory.nbytes / (1024 ** 3) < self.task_bytes:
                    self.data_in_memory = self._update_data(self.data_in_memory, data)
                else:
                    self._to_file(self.data_in_memory.to_dataset('features'),
                                  self.output_path_stats,
                                  mode='a')
                    self.data_in_memory = None

    def archive(self,
                coord_dict: dict[str, np.ndarray],
                pred_dict:  dict[str, np.ndarray],):
        """The actual archive function that updates
           the statistics zarr file by the new batches.

        Parameters
        ----------
        coord_dict         : dict[str, np.ndarray]

        Dictionary of the corresponding coordinates to the
        predicted values. The keys are the coordinate names
        and the values are the coordinate values of the predictions
        
        pred_dict         : dict[str, np.ndarray]

        Dictionary of the the predicted values. The keys are the
        name of the predicted variables and the values are the
        prediction arrays.

        Returns
        -------

        Raises
        ------
        ValueError
            If the specified coordinate keys in the predictions
            dictionary does not match the coordinates of the schema
            that is used to store values into. For example, if the
            keys of the predictions dictionary are
            ['variable1', 'latitude', 'longitude', 'time'] and the
            coordinates of the xarray dataset are
            ['latitude', 'longitude'].

        """

        # check of the coordinates of the predictions matches the
        # specified data_schema
        if not set(coord_dict.keys()) == set(self.out_datafile.coords) - {'stats'}:
            message = 'All coordinates of the data_schema must '
            message += 'be included in the coord_dict.'
            raise ValueError(message)

        # handle the case where we archive the samples separately or all at once
        if self.per_sample:
            for index in range(next(iter(pred_dict.values())).shape[0]):
                data = self._sample_sum_count(coord_dict,
                                              pred_dict,
                                              sample_index=index,)
                self._insert_output(data)
        
        else:
            coord_dict, pred_dict = self._flatten(coord_dict, pred_dict)
            data = self._flatten_sum_count(coord_dict, pred_dict)
            self._insert_output(data)

    def close(self, origin: str = ''):
        """Archive the remaining batches in memory if any and
           create the final output file from the stats zarr and
           remove the stats zarr if asked.
        """

        # archive the remaining batches in memory
        if self.data_in_memory is not None:
            self._to_file(self.data_in_memory.to_dataset('features'),
                          self.output_path_stats,
                          mode='a')
            self.data_in_memory = None
        
        # calculate the average from sum and count to make the
        # final file
        final_ds = self._open_file(self.output_path_stats)
        final_ds = final_ds.sel(stats='sum') / final_ds.sel(stats='count')
        
        # and save the final output file on disk
        message = f'initializing {self.output_path.name} at '
        message += str(self.output_path.parent)
        self.logger.info(message)
        
        self._to_file(final_ds,
                      self.output_path,
                      mode=None)
        
        # remove the stats zarr file if asked
        if not self.keep_stats:
            self._delete_existing(self.output_path_stats)
