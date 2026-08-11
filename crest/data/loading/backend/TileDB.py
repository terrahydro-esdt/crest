from __future__ import annotations
from crest.utils import json_safe
from .BaseBackend import BaseBackend

from dask.diagnostics import ProgressBar
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path

import dask.array as da
import tiledb as tdb
import xarray as xr
import numpy as np
import threading
import sparse
import dask
import json
import time
import sys


# Thread locks to prevent simultaneous block access (which segfaults tiledb)
_TILEDB_LOCKS = defaultdict(threading.Lock)


def _load_sparse_block(uri: str, data: tdb.Array, block_info: dict) -> sparse.COO:
    """Retrieve a block of sparse data from the TileDB dataset. 
    
    Parameters
    ----------
    uri : str
        TileDB file data URI, used to select the proper thread lock.
    data : tiledb.Array
        Array object returned by `tiledb.open(uri)`.
    block_info : dict
        Information about the block to load.

    Returns
    -------
    sparse.COO
        COO array object containing data for the requested block.    
    
    """
    
    assert(None in block_info), block_info
    index = block_info[None]['array-location']
    shape = block_info[None]['chunk-shape']
    names = [dim.name for dim in data.domain]

    # Select slices corresponding to the current block location
    # TileDB appears to segfault here sometimes, which may be
    # due to too many concurrent reads
    # Use a lock for each data block index to avoid segfaults
    count = 0
    while True:
        try:
            with _TILEDB_LOCKS[(uri, tuple(index))]:
                array = data.subarray(tuple(map(slice, *zip(*index))))
            break

        # Retry on error like '[TileDB::S3] Error: Failed to read S3 object'
        except tdb.libtiledb.TileDBError:
            count += 1
            if count >= 5:
                raise
            time.sleep(0.1)

    # Sparse arrays require coordinates to be zero-based 
    coord = [array[dim]-idx[0] for dim,idx in zip(names, index)]
    dtype = array['data'].dtype
    fillv = np.asarray(np.nan if np.issubdtype(dtype, np.floating) 
                       else (-1 if np.issubdtype(dtype, np.number)
                       else ''))
    return sparse.COO(coord, array['data'], shape, fill_value=fillv)


def tiledb_to_xarray(path: Path | str, **kwargs) -> xr.Dataset:
    """Load a TileDB dataset and convert into an xarray Dataset.
    
    Parameters
    ----------
    path : Path | str
        Path to the TileDB location (i.e. the group URI). 
    **kwargs
        Any configuration parameters that are used by `tiledb.Config` 
        (e.g. sm.io_concurrency_level). Used when opening the Array objects 
        contained within the TileDB group.

    Returns
    -------
    xr.Dataset
        A lazy xarray Dataset containing the data in the TileDB dataset.
    
    """
    
    # Calculate the chunk size for every block across all dimensions
    _sizes = lambda d: (d.tile,)*int(d.size//d.tile)+(int(d.size%d.tile),)
    chunks = lambda d: tuple( filter(lambda size: size > 0, _sizes(d)) )

    # Allow tiledb only one thread to prevent interference with dask threading
    config = tdb.Config({
        'sm.compute_concurrency_level' : 1, 
        'sm.io_concurrency_level'      : 1,
        # 'py.init_buffer_bytes'         : 25 * 1024**3,
        # 'sm.memory_budget'             : 25 * 1024**3,
        # 'filestore.buffer_size'        : 25 * 1024**3,
        # # 'sm.partial_tile_offsets_loading' : True,
        # # 'sm.skip_checksum_validation'  : True,
        # # 'py.use_arrow'                 : False, 
        # # 'py.deduplicate'               : 'false',
        # 'sm.enumerations_max_size'     : 25 * 1024**3,
        # 'sm.enumerations_max_total_size' : 30 * 1024**3,
        # 'sm.max_tile_overlap_size'     : 25 * 1024**3,
        # # 'sm.mem.malloc_trim' : False,
        # 'sm.mem.tile_upper_memory_limit' : 25 * 1024**3,
        # 'vfs.max_batch_size' :  2 * 1024**3,
        # 'vfs.s3.max_parallel_ops' : 5,
    } | kwargs)

    coords = {}
    arrays = {}
    xattrs = {}
    
    # Iterate over all Array objects within the TileDB group at the given path
    with tdb.Group(str(path)) as items:
        for item in items:
            data = tdb.open(item.uri, config=config) 
            dims = [d.name for d in data.domain]
            name = item.name
            xattrs[name] = data.meta.get('xarray.attrs', {})
            
            # Coordinates have a single dimension with the same name
            if (len(dims) == 1) and (dims[0] == name):
                coords[name] = data[:]['data']

                # Datetimes stored as int64 due to a tiledb conversion bug
                if name == 'datetime':
                    coords[name] = coords[name].astype('datetime64[ns]')
            
            # All other group items are feature arrays
            else:
                # Get exact chunks if they are stored in the metadata
                chunk = [data.meta.get(d.name, chunks(d)) for d in data.domain]
                dtype = data.attr(0).dtype
                fillv = np.asarray(np.nan if np.issubdtype(dtype, np.floating)
                                   else (-1 if np.issubdtype(dtype, np.number)
                                   else ''))
                
                arrays[name] = (dims, da.map_blocks(**{
                    'chunks' : tuple(chunk),#map(chunks, data.domain)),
                    'func'   : partial(_load_sparse_block, item.uri, data),
                    'meta'   : sparse.COO(
                        [np.empty((0,), dtype=int)] * len(dims), 
                         np.empty((0,), dtype=dtype), 
                        shape=(0,) * len(dims), fill_value=fillv),
                }))
        
        dataset = xr.Dataset(arrays, coords=coords)
        dataset.attrs = json.loads(items.meta.get('xarray.attrs', '{}'))
        for key, attrs in xattrs.items():
            if attrs:
                dataset[key].attrs = json.loads(attrs)
    return dataset
    

class TileDB(BaseBackend):
    """ Backend class to handle reading and writing to TileDB storage """

    def open(self, path: Path|str|None = None, **kwargs) -> xr.Dataset:
        """ Open a TileDB location and return the associated xarray object """
        return tiledb_to_xarray(self.path if path is None else path, **kwargs)


    def cache(self, dest, data: xr.Dataset, stream=sys.stdout, **kwargs):
        """ Write the given xarray Dataset to a TileDB database """
        assert('.tiledb' in str(dest)), f'Not a tiledb path: {dest=}'

        # Long path names can cause segfaults when writing arrays, which
        # can be (partially) addressed by using a temp folder and renaming
        # temp = dest# Path('tmp.tiledb')
        # if temp.exists():
        #     temp.delete()
        #     # shutil.rmtree(temp.as_posix())

        cfg = tdb.Config({
            # 'sm.check_global_order': 'false', 
            #'sm.check_coord_dups': 'false', 
            #'sm.dedup_coords': 'false', 
            #'sm.check_coord_oob': 'false', 
            
            # 20GB / 2GB
            'sm.memory_budget': 2*10737418240,  
            "sm.memory_budget_var": 2*10737418240,
            "filestore.buffer_size": str(20*104857600), 
        })

        # Note: if tiledb is segfaulting when trying to create or write
        #  arrays, it may be due to an import (e.g. tensorflow). Make 
        #  sure tiledb is imported first, before other libraries.
        tdb.group_create(dest.as_posix())
        with tdb.Group(dest.as_posix(), mode='w') as group:
            group.meta['xarray.attrs'] = json_safe(data.attrs, True)
            
            # Add coordinates
            dims = {}
            for dim in data.dims:
                val = data[dim].values
                if np.issubdtype(val.dtype, np.datetime64):
                    val = val.view('int64')

                chunksize = max(data.chunks[dim])
                dimension = dims[dim] = tdb.Dim(**{
                    'name'   : dim, 
                    'domain' : (0, val.size-1), 
                    'tile'   : chunksize, 
                    'dtype'  : np.int64,
                })
                
                domain = tdb.Domain(dimension)
                schema = tdb.ArraySchema(**{
                    'domain' : domain, 
                    'attrs'  : [tdb.Attr(name='data', dtype=val.dtype)],
                    'sparse' : False,
                })

                path = dest.joinpath(dim).as_posix()
                tdb.Array.create(path, schema)
                with tdb.open(path, mode='w') as A:
                    A[:] = val
                    A.meta['xarray.attrs'] = json_safe(data[dim].attrs, True)
                    A.meta['chunks'] = data.chunks[dim]
                group.add(str(dim), str(dim), relative=True)

            # Add features
            jobs = []
            handles = []
            from tqdm import tqdm
            for feature in tqdm(data, file=stream):
                if feature == 'valid_mask': 
                    continue

                values = data[feature]
                domain = tdb.Domain(*[dims[d] for d in values.dims])
                schema = tdb.ArraySchema(**{
                    'domain' : domain, 
                    'attrs'  : [tdb.Attr(name='data', dtype=values.dtype)],
                    'sparse' : True,
                })

                path = dest.joinpath(feature).as_posix()
                tdb.Array.create(path, schema)
                with tdb.open(path, mode='w') as A:
                    A.meta['xarray.attrs'] = json_safe(values.attrs, True)
                    for key, chunks in values.chunksizes.items():
                        A.meta[key] = chunks
                group.add(str(feature), str(feature), relative=True)

                # Summary shouldn't be cached per block (i.e. per stat)
                if feature == 'summary': 
                    continue

                @dask.delayed
                def _write_block(A, blk_idx, block, chunks):
                    if block.nnz:
                        if np.issubdtype(block.data.dtype, np.number):
                            valids = np.isfinite(block.data)
                        else: valids = np.ones_like(block.data, dtype=bool)
                        offset = [[sum(c[:i])] for i,c in zip(blk_idx, chunks)]
                        coords = block.coords[:, valids] + np.array(offset)
                        if valids.any():
                            A[tuple(coords)] = block.data[valids]

                blocks = values.data.blocks
                chunks = values.chunks
                handle = tdb.open(path, mode='w', config=cfg)
                handles.append(handle)
                for blk_idx in product(*map(range, blocks.shape)):
                    jobs.append(_write_block(handle, blk_idx, blocks[blk_idx], chunks))

        # Caching separately enables using the newly cached data for summary
        # generation (which would be faster). Not yet implemented since times
        # using the raw data itself seem reasonable enough for now.
        if 'summary' in data:
            @dask.delayed
            def _write_summary(A, data):
                size = tuple([d.size for d in A.domain])
                rows = np.repeat(np.arange(size[0]), size[1])
                cols = np.tile(np.arange(size[1]), size[0])
                A[rows, cols] = data

            path = dest.joinpath('summary').as_posix()
            handle = tdb.open(path, mode='w')
            handles.append(handle)
            jobs.append(_write_summary(handle, data['summary'].data))
            
        with ProgressBar(out=stream):
            dask.compute(*jobs)
        dask.compute(*[h.close() for h in map(dask.delayed, handles)])
            
        # Run a metadata consolidation now that all data has been cached
        for mode in ['array_meta', 'fragment_meta']:
            config = tdb.Config({"sm.consolidation.mode": mode})
            
            for feature in data:
                path = dest.joinpath(feature)
                if path.exists():
                    tdb.consolidate(path.as_posix(), config=config)