from __future__ import annotations
from .Writer import Writer
from .Indexer import Indexer

from collections import defaultdict as dd
from tempfile import TemporaryDirectory
from itertools import groupby
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import xarray as xr
import numpy as np
import uuid
import os

try:
    Int = int | np.int32 | np.int64
except TypeError:
    Int = int  # fallback when numpy is mocked (e.g. during doc generation)

class StageWriter(Writer):
    """ Class that stages data in parquet fragments for later consolidation.

    Notes
    -----
    Staging (i.e. calling StageWriter.stage_many_coords and .flush) is process
    safe. Multiple StageWriters can stage data within the same directory, with
    a single ZarrWriter being used to occasionally consolidate into the zarr.
    
    Parameters
    ----------
    data_schema : xr.DataArray
        The data schema that any archived data will adhere to. This should
        contain, at minimum, the coordinates to use when creating the zarr.
    stage_path  : str | Path | None
        Path that the staged data should be written to. If None (default),
        a temporary directory is created and used, which is automatically
        cleaned up once the writer is closed.
    max_queue   : int
        The number of batches to collect before writing data held in memory
        into parquet fragments on disk.         
    n_buckets   : int
        The number of hash buckets used to stage data. Enables the randomized
        stream of data to be organized into many localized streams which are
        able to be written more efficiently, as well as allowing parallelism.
    strict_dims : list[str]
        Coordinate dimensions for which values need to be matched exactly
        rather than writing to the nearest schema value. For example, if
        strict_dims=['datetime'] then any archived datetime values must
        match exactly with datetimes that are contained in the schema; but
        e.g. latitude values would not need to match exactly, and would
        instead be written to the nearest value contained within the schema.
    **kwargs
        Additional keyword arguments are discarded.
        
    """
    
    def __init__(self,
        data_schema : xr.DataArray,
        stage_path  : str | Path | None = None,
        max_queue   : int = 100000,
        n_buckets   : int = 128,
        strict_dims : list[str] = ['datetime'],
        **kwargs,
    ):
        self.schema = data_schema
        self.max_queue = max_queue
        self.n_buckets = min(n_buckets, np.prod(self.numblocks))
        self._stage_path = stage_path

        # Create helpers for mapping coord -> index
        self._indexers = {dim: Indexer(**{
            'coords' : self.coords[dim].values,
            'name'   : dim,
            'mode'   : 'strict' if dim in strict_dims else 'nearest',
        }) for dim in self.dims} 


    @property
    def is_open(self) -> bool:
        """ Whether the writer is currently open """
        return hasattr(self, '_directory')
        
    
    def open(self):
        """ Open the writer and initialize all buffers """
        assert(not self.is_open), f'{self} is already open'
        
        # Create the temporary staging directory
        if self._stage_path is None:
            self._directory = TemporaryDirectory()
        else: self._directory = Path(self._stage_path)
        create = lambda p: p.mkdir(parents=True, exist_ok=True)
        create(self.stage_path)

        # Create the bucket buffers and their directories
        self._counter = dd(int)
        self._buffers = {b: dd(list) for b in range(self.n_buckets)}
        for b in self._buffers:
            create(self.stage_path.joinpath(f'bucket-{b:04d}'))
            
                
    def close(self):
        """ Close the writer, flushing any pending data to disk """
        if self.is_open:
            self.flush()
            if hasattr(self._directory, 'cleanup'):
                self._directory.cleanup()
            self.__dict__.pop('_directory')
            self.__dict__.pop('_counter')
            self.__dict__.pop('_buffers')

    
    @property
    def stage_path(self):
        """ Path to the staging area for intermediate data storage """
        assert(self.is_open), f'{self} is not open'
        if isinstance(self._directory, TemporaryDirectory):
            return Path(self._directory.name)
        return self._directory 

    
    def stage_many(self,
        indices : dict[str, np.ndarray],  
        values  : dict[str, np.ndarray],
    ) -> None:
        """ Stage a batch using integer indices """
        coords, indices = zip(*indices.items())
        indices = np.stack(indices, axis=1)

        # Perform sanity checks on the given dicts
        assert(indices.ndim == 2), f'Expected 2 dimensions: {indices.shape=}'
        assert(sorted(coords) == sorted(self.dims)), f'Missing dims: {coords}'
        assert(all(v.shape == (len(indices),) for v in values.values())), \
            f'Unexpected shapes: { {k:v.shape for k,v in values.items()} }'

        # Group indices into buckets by hashing on their block index
        ks, vs = zip(*values.items())
        ix_val = zip(indices // np.array(self.chunksize), *indices.T, *vs)
        bucket = lambda iv: hash(tuple(iv[0])) % self.n_buckets
        for b, group in groupby(ix_val, bucket):
            ix, *vals = zip(*group) 

            # Increment the bucket counter and extend its buffers
            self._counter[b] += len(ix)
            for k, v in zip(coords + ks, vals):
                self._buffers[b][k].extend(v)

        # Flush to parquet fragments if enough items have been accumulated
        if sum(self._counter.values()) >= self.max_queue:
            self.flush()

    
    def stage_many_coords(self,
        coords : dict[str, np.ndarray], 
        values : dict[str, np.ndarray],
    ) -> None:
        """ Stage a batch using real coordinates """
        # Ensure all given arrays are actually vectors
        coords = {k: c.ravel() for k,c in coords.items()}
        values = {k: v.ravel() for k,v in values.items()}

        # Verify all coordinate dimensions are present
        if not all(d in coords for d in self.dims):
            raise ValueError(f'Expected {self.dims}, found {list(coords)}')

        # Verify coordinate vectors are all the same size
        lengths = set(map(len, coords.values()))
        if len(lengths) != 1:
            raise ValueError(f'Invalid coordinate vectors: {lengths=}')

        # Map real coordinates to their respective indices
        indices = {d: self._indexers[d](coords[d]) for d in self.dims}
        self.stage_many(indices, values)


    def flush(self) -> None:
        """ Flush each non-empty bucket to a parquet fragment """
        for b, count in self._counter.items():            
            if count: 
                array = {k: pa.array(v) for k,v in self._buffers[b].items()}
                table = pa.table(array)
    
                bucket_dir = self.stage_path / f'bucket-{b:04d}'
                bucket_hex = uuid.uuid4().hex
    
                # Ensure atomic write
                tmp = bucket_dir / f'.part-{bucket_hex}.parquet'
                out = bucket_dir / f'part-{bucket_hex}.parquet'
                kws = {'compression': 'zstd', 'write_statistics': False}
                pq.write_table(table, tmp, **kws)
                os.replace(tmp, out)

                # Clear the buffer data that was just written
                self._counter[b] = 0
                for v in self._buffers[b].values():
                    v.clear()

