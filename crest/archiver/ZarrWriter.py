from crest.utils import S3Path, silence_warnings
from .Writer import Writer
from .StageWriter import StageWriter

from zarr.convenience import consolidate_metadata
from typing import Tuple
from pathlib import Path
from shutil import rmtree 

import pyarrow.dataset as pads
import xarray as xr
import numpy as np
import zarr


class ZarrWriter(Writer):
    """ Class that reads staged data and writes to zarr.

    Notes
    -----
    Writing (i.e. calling ZarrWriter.flush) is not thread or process safe.
    
    Parameters
    ----------
    data_schema  : xr.DataArray
        The data schema that any archived data will adhere to. This should
        contain, at minimum, the coordinates to use when creating the zarr.
    output_path : Path | S3Path
        Path that the output zarr should be written to.
    stage_writer : StageWriter
        The StageWriter object that is being used to stage data. Solely used
        to determine the staging directory, since the location is unknown
        until the writers have been opened (if using a temporary directory).
    delete_extra : bool
        To allow iteratively updating the output zarr during archiving, two
        additional variables are created per feature: f'{feature_name}__sum'
        and f'{feature_name}__count'. If `delete_extra=True`, rather than 
        retaining them in the final zarr (thus using 3x larger space on disk),
        all of these extra variables will be deleted once the writer is closed.
    
    """
    
    def __init__(self,
        data_schema  : xr.DataArray,
        output_path  : Path | S3Path,
        stage_writer : StageWriter,
        delete_extra : bool = True,
    ):
        self.schema = data_schema
        self.output_path = output_path
        self.stage_writer = stage_writer
        self.delete_extra = delete_extra


    @property
    def is_open(self) -> bool:
        return hasattr(self, '_zarr_f')
        
    
    def open(self):
        assert(not self.is_open), f'{self} is already open'
        
        # Delete the output if it already exists
        if self.output_path.exists():
            if hasattr(self.output_path, 'delete'):
                self.output_path.delete()
            else: rmtree(self.output_path.as_posix())
        
        # Open the zarr and create state variables
        self._zarr_f = zarr.open(self.output_path, mode='a')
        self._arrays = {}
        self.total_rows = 0
        
    
    def close(self):
        if self.is_open:
            # Flush all remaining staged data to disk
            while self.flush(): pass

            # Delete extra variables if requested
            if self.delete_extra:
                for name in self._arrays:
                    for key in self._extra_keys(name):
                        if key in self._zarr_f:
                            del self._zarr_f[key]
                consolidate_metadata(self.output_path)
                
            # Reset object attributes
            self.__dict__.pop('_zarr_f')
            self.__dict__.pop('_arrays')
            total, self.total_rows = self.total_rows, 0
            return total
        return 0
        
    
    def flush(self, max_bucket_files: int=400, delete_after: bool=True) -> int:
        """ Scan staged parquet fragments, aggregate, and write to zarr.

        Notes
        -----
        Writing to the zarr is performed in four steps:
        1. Group rows by chunk
        2. For each chunk, load that chunk once
        3. Apply all updates to that chunk in memory
        4. Writes that chunk once, back into the zarr

        Zarr has to decompress, modify, and recompress an entire chunk whenever
        any part of that chunk is modified. The steps above minimize the IO and
        decompress/recompress cycles so that they need to only be performed one
        time for any chunk during a batch update. 
        
        """
        files = self._list_bucket_files(max_bucket_files)
        if not files:
            return 0
            
        # Build a Dataset over the selected files
        dataset = pads.dataset([str(f) for f in files], format='parquet')
        if hasattr(dataset, 'scanner'):
            dataset = dataset.scanner()
        
        total_rows = 0
        for batch in dataset.to_batches():
            if batch.num_rows == 0: continue
            total_rows += batch.num_rows

            # Load updates and determine the blocks that need to be pulled
            columns = dict(batch.to_pydict())
            indices = np.stack([columns.pop(d) for d in self.dims], axis=1)
            blk_idx = indices // np.array(self.chunksize)
            blk_ids = np.ravel_multi_index(blk_idx.T, self.numblocks)
            
            # Order rows by block id to maximize write locality
            ordered = np.argsort(blk_ids)
            indices = indices[ordered]
            blk_idx = blk_idx[ordered]
            blk_ids = blk_ids[ordered]
            feature = {k: np.array(v)[ordered] for k,v in columns.items()}

            # Find boundaries where chunk ids change
            starts = np.where(np.diff(blk_ids, prepend=-1) != 0)[0]
            finish = np.concatenate([starts[1:], np.array([len(blk_ids)])])
            
            # Helpers to pull a block region out of the full zarr array
            region = lambda c, shp, chk: slice(c*chk, min(shp, (c+1)*chk))
            slices = lambda i: (*map(region, i, self.shape, self.chunksize),)
            astype = lambda v: v.astype(self.dtype)

            # 1. Iterate over each block we need to update
            for s, e in zip(starts, finish):

                # Get the local indices for the block, its shape, and size
                block = slices(blk_idx[s])
                local = indices[s:e] - np.array([s.start for s in block])
                shape = tuple(s.stop - s.start for s in block)
                n_ele = int(np.prod(shape))

                # Calculate the flattened index and number of updates for each
                index = np.ravel_multi_index(local.T, shape)
                count = np.bincount(index, minlength=n_ele)
                touch = np.nonzero(count)[0]
                
                # Update global average variables for each feature
                for name, value in feature.items():
                    
                    # Calculate the sum total of the update for each index
                    total = np.bincount(index, value[s:e], minlength=n_ele)

                    # 2. Load the full block region to update
                    arrays, totals, counts = self._get_feature(name)
                    blk_total = totals[block].reshape(-1)
                    blk_count = counts[block].reshape(-1)
                    blk_array = arrays[block].reshape(-1)

                    # 3. Apply updates to the block in memory
                    blk_total[touch] += astype(total[touch])
                    blk_count[touch] += astype(count[touch])
                    blk_array[touch] = astype(blk_total[touch]/blk_count[touch])

                    # 4. Write the updated block back into its region
                    totals[block] = blk_total.reshape(shape)
                    counts[block] = blk_count.reshape(shape)
                    arrays[block] = blk_array.reshape(shape)

        # Cleanup consumed files
        if delete_after:
            for f in files:
                try:    f.unlink()
                except: pass
        consolidate_metadata(self.output_path)
        self.total_rows += total_rows
        return total_rows
        
                    
    def _get_feature(self, name: str) -> list:
        """ Return zarr arrays for a feature, creating them if necessary """
        assert(self.is_open), f'{self} is not open'
        as_nan = silence_warnings(lambda T: np.array(np.nan).astype(T).take(0))
        if name not in self._arrays:
            
            # Create the coordinates if they don't exist
            for dim, size, chunk in zip(self.dims, self.shape, self.chunksize):
                if dim not in self._zarr_f:
                    values = self.coords[dim].values
                    coords = self._zarr_f.create(**{
                        'name'   : dim,
                        'shape'  : (size,),
                        'chunks' : (min(chunk, size),),
                        'dtype'  : values.dtype,
                        'fill_value' : as_nan(values.dtype),
                    })
                    coords.attrs['_ARRAY_DIMENSIONS'] = [dim]
                    coords[:] = values
                    
            # Create the feature
            array = self._zarr_f.create(**{
                'name'   : name,
                'shape'  : self.shape,
                'chunks' : self.chunksize,
                'dtype'  : self.dtype,
                'fill_value' : as_nan(self.dtype),
            })
            array.attrs['_ARRAY_DIMENSIONS'] = list(self.dims)

            # Create the feature accumulators
            arrays = [array]
            for key in self._extra_keys(name):
                arrays.append( self._zarr_f.create(**{
                    'name'   : key,
                    'shape'  : self.shape,
                    'chunks' : self.chunksize,
                    'dtype'  : self.dtype,
                    'fill_value' : self.dtype.type(0),
                }) )
                arrays[-1].attrs['_ARRAY_DIMENSIONS'] = list(self.dims)
            self._arrays[name] = arrays
        return self._arrays[name]


    def _extra_keys(self, name: str, suffixes=['sum', 'count']) -> list[str]:
        return [f'{name}__{suffix}' for suffix in suffixes]

    
    def _list_bucket_files(self, max_files_per_bucket: int) -> list[Path]:
        files = []
        for p in sorted(self.stage_writer.stage_path.glob('bucket-*')):
            parts = sorted(p.glob('part-*.parquet'))
            files.extend(parts[:max_files_per_bucket])
        return files
