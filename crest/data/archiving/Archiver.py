from __future__ import annotations
from crest.utils import S3Path
from .StageWriter import StageWriter
from .ZarrWriter import ZarrWriter
from .Writer import Writer

from pathlib import Path
import xarray as xr
import numpy as np
import logging

log = logging.getLogger(__name__)

class Archiver(Writer):
    """ Allows writing batches of data to a zarr on disk.

    Notes
    -----
    This class efficiently handles writing random batches of data by splitting
    the work into two steps: batches are first staged into buckets of parquet
    files, and then are written to zarr once a sufficient number of batches
    have been staged.

    The staging process is process safe, and so multiple StageWriters can stage
    (i.e. StageWriter.stage_many_coords and .flush) simultaneously if desired.
    Writing to zarr is not thread or process safe, however, and so the writing
    function (i.e. ZarrWriter.flush) can be called by only one thread/process.

    Parameters
    ----------
    data_schema : xr.Dataset | xr.DataArray
        The data schema that any archived data will adhere to. This should
        contain, at minimum, the coordinates to use when creating the zarr.
    output_path : str | Path | S3Path
        Path that the output zarr should be written to.
    stage_path  : str | Path | None
        Path that staging files and folders should be written to. If None
        (default), a temporary directory is used, which is automatically
        cleaned up once finished.
    chunksizes  : dict[str, int]
        Chunksizes can be given as a dictionary of coordinate dimensions. If
        unspecified, the chunk scheme defined by the data schema will be used
        instead (or 'auto' chunking if not defined by the data schema).
    write_every : int
        Controls how many batches to stage before writing to the output zarr;
        e.g. write_every=1000 means that every 1000 calls to Archiver.archive,
        the output zarr on disk will be updated with the available staged data.
    overwrite   : bool
        Allow removing an already existing zarr at the output path.
    verbose     : bool
        Whether information should be printed when writing to the output zarr.
    **kwargs
        Additional keyword arguments are used during writer initialization. See
        StageWriter and ZarrWriter class docstrings for available options.

    """

    def __init__(self,
        data_schema : xr.Dataset | xr.DataArray,
        output_path : str | Path | S3Path,
        stage_path  : str | Path | None = None,
        chunksizes  : dict[str, int] | None = None,
        write_every : int = 1000,
        overwrite   : bool = False,
        verbose     : bool = False,
        **kwargs
    ):
        # Make a Path-like object
        if isinstance(output_path, str):
            if output_path.startswith('s3://'):
                output_path = S3Path(output_path)
            else: output_path = Path(output_path)

        # Verify we can overwrite if it already exists
        assert(overwrite or not output_path.exists()), \
            f'{output_path=} exists and {overwrite=}'

        # Get a Dataset object if a DataArray with a features dim was given
        if (
            isinstance(data_schema, xr.DataArray)
            and 'features' in data_schema.dims
        ):
            data_schema = data_schema.to_dataset('features')

        # Get a DataArray object if a Dataset was given
        if isinstance(data_schema, xr.Dataset):
            # Create a dummy variable if none currently exist
            if len(data_schema):
                data_schema = data_schema[next(iter(data_schema))]
            else:
                data_schema = xr.DataArray(0., data_schema.coords)

        # Determine chunksizes to use for the output zarr
        if chunksizes or (not data_schema.chunksizes):
            data_schema = data_schema.chunk(chunksizes or 'auto')

        self.stage_writer = StageWriter(**({
            'data_schema' : data_schema,
            'stage_path'  : stage_path
        } | kwargs))
        self.zarr_writer = ZarrWriter(**({
            'data_schema' : data_schema,
            'output_path' : output_path,
            'stage_writer': self.stage_writer,
        } | kwargs))
        self.schema = data_schema
        self.verbose = verbose
        self.write_every = write_every
        self.output_path = output_path
        self.stage_path = stage_path


    def open(self):
        """ Opens the StageWriter and ZarrWriter """
        self.stage_writer.open()
        self.zarr_writer.open()
        self.batch_count = 0


    def close(self):
        """ First flush any remaining batches for staging, then close """
        self.stage_writer.flush()
        rows = self.zarr_writer.close()
        self.stage_writer.close()

        # A sample can be written multiple times (equaling the average over all
        # its writes), so the total may be > actual number of items in the zarr
        if self.verbose:
            log.debug(f'{self.output_path}: Wrote {self.batch_count:,} ' +
                      f'batches and {rows:,} total samples')
        self.batch_count = 0


    def archive(self,
        coords  : dict[str, np.ndarray],
        values  : dict[str, np.ndarray],
        max_pq  : int = 400,
    ) -> int:
        """ Top-level function for archiving data.

        Parameters
        ----------
        coords : dict[str, np.ndarray]
            Dictionary containing coordinate names as the keys, and coordinate
            vectors as the values. Note that if multi-dimensional arrays are
            given instead of vectors, they will be flattened and treated as
            individual samples. All arrays must be the same size.
        values : dict[str, np.ndarray]
            Dictionary containing feature names as the keys, and feature
            vectors as the values. Note that if multi-dimensional arrays are
            given instead of vectors, they will be flattened and treated as
            individual samples. All arrays must be the same size.
        max_pq : int
            Maximum number of parquet files to pull from each bucket when
            writing to zarr.

        Returns
        -------
        int
            Total number of items that have been written to the output zarr.

        """
        self.stage_writer.stage_many_coords(coords, values)
        self.batch_count += 1
        if (self.batch_count % self.write_every) == 0:
            rows = self.zarr_writer.flush(max_pq)
            if self.verbose:
                log.info(f'\nWrote {rows:,} samples to {self.output_path} ' +
                         f'| Total: {self.zarr_writer.total_rows:,}')
        return self.zarr_writer.total_rows
