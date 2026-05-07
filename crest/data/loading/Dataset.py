from collections.abc import Iterable, Callable, Collection
from collections import defaultdict as dd
from dask.diagnostics import ProgressBar
from fsspec.mapping import FSMap 
from contextlib import nullcontext, redirect_stdout
from functools import partial, cached_property
from itertools import starmap
from pathlib import Path
from numbers import Number, Integral as Int 
from logging import Logger 
from typing import Union 
import re
from pprint import pprint

import cloudpickle as pkl
import xarray as xr
import numpy as np 
import dask.array as da
import dask
import logging
import io 

from crest.utils import Stopwatch, optimize_blocks, S3Path
from crest.base import BaseSet
from crest.nodes.tensorflow import Node as TFNode
from crest.model import HierarchalTensorGraph
from crest.model import IOSpec
from .Datafile import Datafile
from .Blockset import Blockset


class Dataset(BaseSet):
    """Wraps a collection of Datafiles, thus loading multiple sources. 
    
    Notes
    -----
    Any attributes created in __init__ (or in general, prior to 
    Dataset.generate_samples) may not be available once generate_samples
    is actually called. This is due to only pickling attributes initially
    given to __init__, for both Dataset and Datafile. If modifications 
    need to be made to the Dataset object or underlying Datafile objects,
    they should be applied during the generate_samples function.

    Parameters
    ----------
    locations : Iterable[Datafile | Path | str | FSMap]
        The set of Datafiles or data locations which form this 
        Dataset. If any paths are passed in via this parameter,
        a new Dataset object is created to load from that path.
        Because this inherits from BaseSet, functions existing 
        in the Datafile class can be called by this object in 
        order to apply the function across all Datafiles in this 
        set. See crest.crest.base.BaseSet for more details.
    **kwargs
        Any additional keyword arguments are passed into each
        newly created Datafile, thus allowing global options
        to be set across Datafiles when creating the Dataset.
        Note that this is only used for locations which are not
        already passed in as a Datafile object.
    
    """
    def __init__(self, locations: Iterable[Union[Datafile, Path, str, FSMap, S3Path]], **kwargs):
        self.container = list(map(partial(Datafile.load, **kwargs), locations))
        assert(len(self)), 'Must pass at least one object to init Dataset'
        for i, datafile in enumerate(self): datafile.dataset_index = i
        

    @cached_property
    def dtype(self):
        """ Sample dtype """
        return np.dtype([(f'Data_{i}', df.dtype) for i,df in enumerate(self)])


    @cached_property
    def total_bytes(self) -> float:
        """ Estimate of the total input/output bytes of the data """
        inp_bytes = sum(self.data.nbytes)
        itemsizes = self.dtype.itemsize
        out_bytes = itemsizes * np.prod(self.shape.ix[:-1], 1, dtype=float)
        assert((out_bytes > 0).all()), f'Arrays are too large to compute bytes'
        return inp_bytes + out_bytes.sum()


    @cached_property
    def logger(self):
        """ Create a logging object """
        logger = logging.getLogger('Dataset')
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        return logger


    def summaries(self, compute=True, keep='last') -> xr.DataArray:
        """ Gather summary statistics of all component Datafiles """
        summary = self.summary.map_blocks(xr.DataArray.as_numpy)
        summary = xr.concat(summary, dim='features', join='outer')
        # for s,label in zip(summary,self.key_label):
        #     s['features'] = [f'{label}>>{k}' for k in list(s.features.values)]
        # summary = xr.concat(summary, dim='features')
        summary = summary.drop_duplicates('features', keep=keep)
        if compute:
            with ProgressBar():
                print(f'{summary=}\n\nComputing summaries...')
                summary = summary.compute() 
        return summary


    def get_feature(self, feature: str) -> xr.DataArray:
        """ Return an xr.DataArray containing the requested feature's data """
        available = []
        for array in self.data:
            if feature in array.features:
                return array.sel(features=feature, drop=True)
            available.append(array.features)
        else: raise ValueError(f'{feature=} not found: {available=}')
            

    def align(self, a: str, *b, n=20):
        """ Helper used to align log text """
        return f'{a:>{n}}: '+' '.join(map(str, b))


    def generate_samples(self, 
        block_bytes : Number = 1e9,
        numblocks   : Union[int, Collection[int]] = 0,
        compute     : bool = True,
        verbose     : bool = True,
        optimize    : bool = True,
        shuffle     : bool = False,
        return_objs : bool = False,
        logger      : Union[Logger, None] = None,
        log_level   : Union[int, None] = None,
        save_path   : Union[str, Path, None] = None,
        cache_path  : Union[Path, S3Path, None] = None,
        overwrite   : bool = False,
    ):# -> da.Array | Iterator[Delayed]:
        """Generate the dask array containing all valid samples.

        Notes
        -----
        Sequence of steps to prepare the data:
            1. Ensure all Datafiles are aware of all dimensions
            2. Determine number of blocks per dimension and element overlaps
            3. Rechunk data based on the required number of blocks
            4. Apply overlap, create Blocksets with one Block per Datafile

        Sequence of steps to find valid samples (in parallel across Blocksets):
            1. Find valid windows for each Block (based on Block.valid_percent)
            2. Match windows between Blocks that are within a certain distance
            3. Create a lazy dask array containing indices of matched windows
            4. Combine arrays from all Blocksets into the final dask array

        Parameters
        ----------
        block_bytes : Number
            Number of bytes that should be allocated to each block and worked
            on in parallel (default=1e8; 100MB). Note that this is just a proxy
            for the amount memory that will be used when computing a block, as 
            the actual amount used is dependent upon the number of matches that
            are found in the block and thus can vary significantly. 
        numblocks : int 
            Alternative to giving a block_bytes value. If numblocks > 0, the 
            requested number of blocks is the target block total. While the 
            exact number of blocks is not always possible to create, an attempt
            is made to get as close as possible to the requested value.
        compute   : bool
            Whether the lazy dask array of Sample objects should be computed 
            and returned, or just a list of the `dask.delayed.Delayed` tasks 
            created for all blocks (with one block per task). If the list of 
            task objects is returned with `compute=False`, the `.compute()` 
            method can be called on each to independently compute the blocks. 
        verbose   : bool
            Whether logs should be shown when preparing and generating samples.
        optimize  : bool
            Whether dask.optimize should be used on the full task graph. This
            can speed up access when batching data, but incurs a higher cost
            when initially generating sample blocks. 
        shuffle   : bool
            To the extent possible, shuffle sample ordering.
        logger    : Callable
            Logging function used when `verbose=True`. This is `print` by 
            default, but an actual logging function like `logging.info` can be
            given instead.
        log_level : int | None
            Log level that should be used by the logger, e.g. logging.INFO. If
            None is given, the default level set by the logger is used.  
        save_path : str | Path | None
            Save the generated sample array to a pickle file at the given path.
        cache_path : Path | S3Path | None
            Cache the generated blocks at the given path. Caching blocks is
            distinct from caching Datafiles, as this parameter saves data in
            the format used by the Batcher in order to speed up batching.
        overwrite : bool
            Whether the cache_path blocks should be overwritten, if they exist. 
            
        Returns
        -------
        dask.array.Array | list[dask.delayed.Delayed]
            Either the lazy array which contains all valid samples which were 
            found (if `compute=True`, the default); or a list of the Delayed
            task objects that correspond to the individual block computations.

        """
        if logger is not None:
            self.__dict__['logger'] = logger
        if log_level is not None:
            self.logger.setLevel(log_level)
            
        # In order to avoid overlapping logs with multiple processes,
        # we accumulate all log text and log only once at the end
        log_sep = ''.join(['_']*60) + '\n'
        dat_sep = ''.join(['-']*60)
        log_txt = f'\n{log_sep}\nCurrent data:\n{dat_sep}'
        for c in self.container:
            log_txt += f'\n\n{c.data}'
        log_txt += f'\n\n{dat_sep}\n\nPreparing data...\n'
        redirect = redirect_stdout# if cache_path is None else nullcontext

        # if cache_path is not None and verbose:
        #     self.logger.info(log_txt)
            
        # Capture stdout to log appropriately
        try:     
            with redirect(io.StringIO()) as buffer:
                # Ensure all Datafiles are aware of all dimensions
                # i.e. create virtual dimensions where necessary
                self.ensure_dims( set.union(*map(set, self.dims)) )

                # Ensure the coordinate extents are aligned across Datafiles
                # i.e. so that blocks will refer to the same coordinate areas
                # Unfortunately, still not working properly / requires large
                # amount of memory for certain datasets
                # if len(self) > 1: self.ensure_extents()

                # Create the delayed sample blocks
                samples = self.create_blocks(
                    block_bytes = block_bytes,
                    numblocks   = numblocks,
                    verbose     = verbose,
                    optimize    = optimize,
                    shuffle     = shuffle,
                    return_objs = return_objs,
                    cache_path  = cache_path,
                    overwrite   = overwrite,
                )
        except: 
            verbose = True
            raise

        finally: 
            # Log the accumulated text prior to starting block computation
            # if verbose: 
            message = (log_txt+buffer.getvalue())# if cache_path is None else ''
            self.logger.info(message + log_sep)

        # Find all samples in parallel across the created blocksets
        if compute:
            with nullcontext() if not verbose else ProgressBar():
                if verbose: self.logger.info('\nFinding all valid samples...')
                samples = da.hstack( da.compute(*[s() for s in samples]) )
            
            if verbose: self.logger.info(f'\nFound {len(samples):,} samples')
            if save_path is not None:
                Dataset.save(samples, save_path)
        return samples 


    def ensure_extents(self):
        """ Ensure all Datafile extents represent equivalent bounding boxes.

        A universal extent is found by calculating the maximal intersection 
        of all Datafile extents. This is then set as the new extent for each
        Datafile using its nearest respective coordinates, with an additional 
        1-element buffer included at edges (if data is available) to allow for 
        potential sample matches at coordinate extrema. These new Datafile 
        extents will always generate the same intersecting area and so this
        function is idempotent. 

        Still uncertain whether this should be performed automatically (risking
        data being unexpectedly excluded); or left up to the user to correctly
        bound their data (allowing the potential for unnecessary data that 
        slows down sample generation, or possibly a bad block schema).

        """

        def get_extents(data):
            # Handles datetime: https://github.com/pydata/xarray/issues/3256
            return {dim: (
                data[dim].min().to_numpy().min(),
                data[dim].max().to_numpy().max(),
            ) for dim in data.coords if not data[dim].isnull().any()}

        def print_extent(extent, prefix='\t\t'): 
            """ Printing helper """
            for dim in list(extent): 
                print(f'{prefix}{dim}: ({extent[dim][0]}, {extent[dim][1]})')

        # Iterate over all Datafiles to find the global extent
        extent = {}
        for df in self:

            # Get the full extent available in the zarr
            raw = df._raw_data
            ext = get_extents(raw)
            for dim, raw_ext in ext.items():

                # Get the Datafile's subset extent and current global extent
                d_slice = slice(*df.extent.get(dim, raw_ext))
                sub_ext = get_extents(raw[dim].sel({dim: d_slice}))[dim]
                gbl_ext = extent.get(dim, raw_ext)

                # Store the running maximin/minimax extent for each dimension
                extent[dim] = (
                    max(raw_ext[0], sub_ext[0], gbl_ext[0]),
                    min(raw_ext[1], sub_ext[1], gbl_ext[1]),
                )

        print(f'\n\tSetting new global extent:')
        print_extent(extent)

        # Update each Datafile extent using the found global extent
        for df in self: 
            raw = df._raw_data
            ext = {}

            # Generate new Datafile extent for each dimension
            for dim in df.dims:
                vals = raw[dim]

                # Skip any virtual dimensions
                if not vals.isnull().any():
                    minim, maxim = extent[dim]

                    # Find nearest coords, taking the element one index outside
                    find_idx = lambda v, side: np.searchsorted(vals, v, side)
                    in_bound = lambda index: max(0, min(index, len(vals)-1))
                    to_coord = lambda index: vals[in_bound(index)].to_numpy()

                    # +/-1 gives nearest, +/-2 gives one index further outside
                    ext[dim] = ( to_coord( find_idx(minim, 'right')-2 ).min(),
                                 to_coord( find_idx(maxim, 'left' )+2 ).max() )

            # Set the final extent and ensure Datafile.data is regenerated
            print(f'\n\tSetting new extent for {df}:')
            print_extent(ext)
            df.extent = ext
            df.__dict__.pop('data', None)


    def create_blocks(self, 
        block_bytes : Number = 1e8,
        numblocks   : Union[int, Collection[int], None] = 0,
        verbose     : bool = False,
        optimize    : bool = True,
        shuffle     : bool = False,
        return_objs : bool = False,
        cache_path  : Union[Path, S3Path, None] = None,
        overwrite   : bool = False,
    ) -> Union[list, None]:
        """Block data into the requested configuration.

        Parameters
        ----------
        block_bytes : Number
            Number of bytes that should be allocated to each block and worked
            on in parallel (default=1e8; 100MB). Note that this is just a proxy
            for the amount memory that will be used when computing a block, as 
            the actual amount used is dependent upon the number of matches that
            are found in the block and thus can vary significantly. 
        numblocks : int 
            Alternative to giving a block_bytes value. If numblocks > 0, the 
            requested number of blocks is the target block total. While the 
            exact number of blocks is not always possible to create, an attempt
            is made to get as close as possible to the requested value.
        verbose : bool
            Whether logs should be shown when preparing and generating samples.
        optimize : bool
            Whether dask.optimize should be used on the full task graph. This
            can speed up access when batching data, but incurs a higher cost
            when initially generating sample blocks. 
        shuffle : bool
            To the extent possible, shuffle sample ordering.
        return_objs : bool
            If set to True, a list of (delayed) Blockset objects will be
            returned. Otherwise, a list of (delayed) functions to compute 
            matching samples will be returned. 
        
        Returns
        -------
        list
            List of delayed Blockset.find_matches functions.

        """

        if not isinstance(numblocks, (int, type(None))): 
            constrain = [n == 1 for n in numblocks]
        else: constrain = None 

        # Attempt to automatically block the data based on total block count
        if isinstance(numblocks, int):
            self.autochunk(block_bytes, numblocks, constrain, verbose)
            blocks = self.numblocks[0][:-1]

        # Either block the data to an exact number per dimension (e.g. [1,5,9])
        # or try to automatically find a reasonable common blocking scheme 
        # based on the scheme currently used across all Datafiles (if None)
        else:
            blocks = numblocks or np.gcd.reduce(self.numblocks, axis=0)[:-1]

            # Attempt to automatically determine a target block size
            # TODO: fix issue when window_depth > elements per block
            if (numblocks is None) and (max(blocks) == 1) and (max(self.size) > 100):
                blocks  = [1] * len(blocks)
                n_dim   = min(len(blocks), 2)
                maxim   = np.min(self.shape, axis=0)[:-1]
                indices = np.argpartition(maxim, -n_dim)[-n_dim:]

                def max_block(shapes, start=None):
                    """ Return the maximum valid block size for all shapes """
                    check = lambda n: np.ceil(shapes / np.ceil(shapes / n)) == n
                    if start is None:
                        start = int(shapes.min())
                        if check(start).all(): return start
                    start = start or (np.ceil(shapes/2)).min()
                    while (start > 1) and not check(start).all(): start -= 1
                    return int(start)

                for i in range(n_dim):
                    shapes = np.array([shp[indices[i]] for shp in self.shape])
                    blocks[indices[i]] = max_block(shapes, 25)

            if verbose:
                print(f'\tcurrent blocks: {self.numblocks}')
                print(f'\t target blocks: {blocks}')

            # Update Datafile chunks to create the required number of blocks
            # Assuming optimize is only False during caching, we can allow
            #   Datafiles to be cached with differing block counts
            chunks = self.update_blocks(blocks, verify=optimize)
            if verbose: 
                if any(chunks): print(f'\trechunked with: {chunks}')
                print(f'\t result blocks: {self.numblocks.ix[:-1]}')

        # Validate all chunks are at least as large as requested window size
        max_blocks = np.min(list(self.max_valid_blocks), axis=0).astype(int)
        if any(blk > maxim for blk, maxim in zip(blocks, max_blocks)):
            raise Exception(f'Specified blocks {blocks} are greater than the '+
                    f'maximum valid number of blocks {max_blocks}')

        # This can only be run when not caching, as zarr doesn't allow chunks 
        # with non-uniform sizes. Matching anisotropic grids between Datafiles
        # requires us to have non-uniformly sized chunks however, and so this
        # must be run by any object which is generating matchups, e.g. Batcher
        if optimize and not isinstance(numblocks, int): 
            self.autochunk(numblocks=blocks)

        if verbose: 
            indent = '\n\t\t'
            chunks = lambda df: [f'{d:>10} = {c}' for d,c in zip(df.dims, df.chunks)]
            to_str = lambda df: f'{df}'+f'{indent}  '.join(['']+chunks(df))
            print(self.align('Chunks', indent+indent.join(map(to_str, self))))

        # Remainder can be skipped if not optimizing
        if not optimize: return

        # Verify the data chunks are valid for the requested window sizes
        for df in self:
            chunks = df.chunks[:-1]
            window = [df.window_depth[dim] for dim in df.dims]
            assert(len(chunks) == len(window)), [chunks, window, df.dims]

            # Technically, we only need the chunk on either side to be as large 
            #   as the requested length on that side - i.e. (100, 5) means we 
            #   need chunks on the left side of all chunks to be size >= 100,
            #   and chunks on the right side of all chunks to be size >= 5
            # One additional constraint is that the total chunk needs to have
            #   (left + center + right) * valid_percent >= window total
            # This additional constraint handes the case where there exist only
            #   one or two chunks in total
            for dim_chunks, dim_window, dim in zip(chunks, window, df.dims):
                minim_left, minim_right = dim_window

                # Check left/right side constraint
                for i in range(1, len(dim_chunks)-1):
                    left  = dim_chunks[i-1]
                    right = dim_chunks[i+1]

                    if (left < minim_left) or (right < minim_right):
                        raise Exception(f'Using block size {blocks}, {df} ' +
                            f'(shaped {df.shape[:-1]}) is given chunks along axis '+
                            f'"{dim}" = {dim_chunks} - which is invalid for '+
                            f'the requested window size of {list(dim_window)}')

                # Check total size constraint
                for i, center in enumerate(dim_chunks):
                    left  = dim_chunks[i-1] if i>0 else 0
                    right = dim_chunks[i+1] if i<(len(dim_chunks)-1) else 0

                    if ((left+center+right) * df.valid_percent[dim]) < sum(dim_window):
                        raise Exception(f'Using block size {blocks}, {df} ' +
                            f'(shaped {df.shape[:-1]}) is given chunks along axis '+
                            f'"{dim}" = {dim_chunks} - which is invalid for '+
                            f'the requested window size of {list(dim_window)}')

        # Get the current number of blocks
        chunks  = self.chunks.ix[:-1]
        current = self.numblocks.ix[:-1]
        blocks  = np.array(current[0])
        assert(all(((current == blocks) | self.virtual).map(all))), current

        # Calculate required overlaps for each Datafile
        max_res = np.nanmax(self.max_resolution, axis=0)
        idx_res = np.nanargmax(self.max_resolution, axis=0)
        skipdim = idx_res == np.arange(len(self))[:, None]
        overlap = self.calculate_overlap(max_res, 
            dimension_blks = blocks, # Same blocks shape for all Datafiles
            _map = [skipdim],        # Map Datafiles to respective skipdim
        )

        if verbose: 
            buffer = max(map(len, map(str, self))) + 1
            values = ['']
            for df, dims, over in zip(self, self.dims, overlap, strict=True):
                ov_vals = '   '.join(f'{dims[i]}: {o}' for i,o in over.items())
                values += [f'{str(df):>{buffer}} = {ov_vals}']
            print(self.align('Overlaps', '\n                '.join(values)))

            nbytes = (self.total_bytes / np.prod(blocks)) / 1e6
            nelems = sum(map(np.prod, [list(map(np.mean, c)) for c in chunks]))
            print('\nResults:\n  ' + '\n  '.join([
                f'Split into {np.prod(blocks)} total block(s)',
                f'~{nbytes:,.1f} MB/block'.replace('.0',''),
                f'~{nelems:,.0f} items/block'
            ]))

        # Create a list of Blocksets, where each Blockset 
        # contains exactly one Block from each Datafile
        prepped = self.apply_overlap(
            optimize   = optimize, 
            cache_path = cache_path, 
            overwrite  = overwrite,
            _map   = [overlap],    # Map Datafiles to the overlaps
            _delay = not optimize, # Compute only if optimizing
        )
        create_set = partial(Blockset, logger=self.logger, shuffle=shuffle)
        blocksets  = map(dask.delayed(create_set), zip(*prepped, strict=True))
        return [b if return_objs else b.find_matches for b in blocksets]


    def autochunk(self, 
        block_bytes : Number = 1e8, 
        numblocks   : Union[int, Collection[int]] = 0,
        constrain   : Union[list[bool], None] = None,
        verbose     : bool = False,
    ) -> None:
        """Attempt to automatically chunk/block the data.
        
        This method re-chunks data in order to get as close as possible
        to the requested number of bytes contained in each block. To do
        so, it performs a greedy optimization using a few different
        initial conditions at the boundaries of the search space. 

        The step size for the optimization adheres to the constraint that
        blocks must be an integer multiple of their original sizes, working
        under the assumption that data stored in a remote location (i.e. 
        needing to be downloaded locally before using) will be more efficiently
        handled when the native chunking of the remote data is used. Meaning,
        data contained in a given block will never be partially split across
        multiple remote chunks - the block data will always be fully contained
        within one or more chunks, thus minimizing the amount of data which
        needs duplicated or thrown away for each new block. 

        Parameters
        ----------
        block_bytes : Number
            Number of bytes that should be allocated to each block and worked
            on in parallel (default=1e8; 100MB). Note that this is just a proxy
            for the amount memory that will be used when computing a block, as 
            the actual amount used is dependent upon the number of matches that
            are found in the block and thus can vary significantly. 
        numblocks : int 
            Alternative to giving a block_bytes value. If numblocks > 0, the 
            requested number of blocks is the target block total. While the 
            exact number of blocks is not always possible to create, an attempt
            is made to get as close as possible to the requested value.
        constrain : list[bool] | None
            Allows constraining the generated data blocks along specified
            dimensions to 1. For example, if there are three coordinates
            [datetime, latitude, longitude] and the datetime dimension should
            not be split (i.e. there should only be one datetime block), then
            constrain=[True, False, False] can be passed to signify the first
            dimension is constrained to be 1. 
        verbose   : bool
            Whether logs should be shown when preparing and generating samples.

        """
        if isinstance(numblocks, int):
            # Close (over-)estimate for the targeted number of blocks per dimension
            # (inp bytes + (estimated) out bytes) / block_bytes = number of blocks
            if numblocks>0: tgt_blks = numblocks 
            else:           tgt_blks = int(np.ceil(self.total_bytes / block_bytes))
            if verbose: print(self.align('Target block total', f'{tgt_blks:,}'))

            # Maximum number of blocks the data could theoretically be split into, 
            # while still maintaining the required number of elements along each 
            # dimension to fulfill the requested window size
            # required = lambda d_v, data, size: [np.inf if v else len(data[d])//size[d] for d,v in d_v]
            # data_obj = self._typed_data, self.window_total
            # req_blks = map(required, map(zip, self.dims, self.virtual), *data_obj)
            # max_blks = np.min(list(req_blks), axis=0).astype(int)
            max_blks = np.min(list(self.max_valid_blocks), axis=0).astype(int)
            cur_blks = self.numblocks.ix[:-1]
            assert(np.isfinite(max_blks).all()), f'Invalid max block size: {max_blks}'
            assert(min(max_blks) > 0), f'Requested window larger than data: {self.shape}'

            # Apply any constraints by limiting the maximum block count to 1
            if constrain is not None:
                max_blks[constrain] = 1

            if verbose: 
                print(self.align('Max valid blocks', max_blks))
                print(self.align('Current block sizes', list(cur_blks)))

            # Bin the data into histograms with the specified number of blocks
            # in order to get the final chunk sizes
            blocks = optimize_blocks(cur_blks, tgt_blks, max_blks)
        else: blocks = numblocks
        if verbose: print(self.align('Pre-merged blocks', blocks))
        
        labels = list(zip(*self.dims, strict=True))
        coords = [self._typed_data.ix[c].values for c in labels]
        depths = [self.block_minim.ix[c] for c in labels]
        chunks = list(zip(*map(uniform_chunks, blocks, coords, depths)))
        blocks = {tuple(map(len, c)) for c in chunks}
        assert(len(blocks) == 1), blocks
        if verbose: print(self.align('Created blocks', blocks.pop()))
        self.update_chunks(_map=[chunks])



    @classmethod
    def save(cls, samples: da.Array, filename: Union[str, Path]):
        """Save the given samples array as a pickle file at the requested path.
        
        Note
        ----
        The sample array is pickled and stored in the lazy dask representation
        that is given. Thus, if the lazy array is e.g. 5GB in memory but 18TB
        fully expanded, a 5GB file will be stored and the full samples can be
        expanded into their full representation when loaded later.
        
        Parameters
        ----------
        samples  : dask.Array
            Dask array that is created using generate_samples.
        filename : str | Path
            Location to save the given samples to. 

        """
        with Path(filename).open('wb') as f:
            pkl.dump(samples, f)



    @classmethod
    def load(cls, filename: Union[str, Path]) -> da.Array:
        """Load a dask array from a previously saved pickle file.

        Parameters
        ----------
        filename : str | Path
            Location to load the dask array from.

        Returns
        -------
        dask.Array
            The dask array stored at the given location.

        """ 
        assert(Path(filename).exists()), f'{filename} does not exist'
        with Path(filename).open('rb') as f:
            return pkl.load(f)



    def cache(self, 
        numblocks  : Collection[int], 
        cache_dir  : Union[Path, str, FSMap, S3Path] = 'Cache',
        overwrite  : bool = False, 
        verbose    : bool = False,
        fast_check : bool = False,
    ) -> 'Dataset':
        """Cache all Datafiles in new zarr databases for faster access.

        Parameters
        ----------
        numblocks : Collection[int]
            Number of blocks along each dimension which the data should
            be chunked into.
        overwrite : bool
            Flag indicating whether already cached data should be overwritten.
            If False, metadata (but not the data values themselves) are checked
            against the cached data to verify it is the same. This does not
            verify that the data itself is the same, and so changing e.g. one
            of the preprocessor function definitions, might result in the wrong 
            data being used. 
        cache_dir : Path | str | FSMap | S3Path
            Location for the cached data to be stored. By default, data is 
            cached in `./Cache`. 
        verbose   : bool
            Additional information printed.
        fast_check : bool
            Allow skipping intensive computation if all datafile hashes are
            available in the cache location. Note that this could result in
            unexpected characteristics for the data being loaded (e.g. with
            a different number of blocks than requested). 

        Returns
        -------
        Dataset
            Returns self. 

        """

        # Skip if cache folders for all datafile hashes exist
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        elif isinstance(cache_dir, FSMap):
            cache_dir = S3Path(cache_dir)

        paths = [cache_dir.joinpath(df.cache_name) for df in self]
        if overwrite or not (fast_check and all(p.exists() for p in paths)):
            with Stopwatch(f'Cached {len(self)} Datafiles at {cache_dir}'):

                # # Shift target block midpoints between two blocks so overlaps
                # # only need two reads instead of three
                # if not isinstance(numblocks, (int, type(None))):
                #     numblocks = [n*2-1 for n in numblocks]

                # Rechunk the data first
                self.generate_samples(**{
                    'numblocks'  : numblocks, 
                    'compute'    : False, 
                    'verbose'    : verbose,
                    'optimize'   : False,
                })
                self._cache(overwrite, verbose, cache_dir, _delay=False)
                # self.generate_samples(**{
                #     'numblocks'  : numblocks, 
                #     'compute'    : False, 
                #     'verbose'    : verbose,
                #     'optimize'   : True,
                #     'cache_path' : cache_dir.joinpath('Batcher'),
                # })
            print()


        # Reinitialize Datafiles with their cached data
        # Anything handled by the cache (e.g. extent) can be dropped
        else: 
            self.logger.info(f'All caches already exist ({fast_check=})')
            self.reset(**{
                '_kwmap'        : {'location': paths},
                'region'        : None,
                'extent'        : {}, 
                'preprocessors' : [],
            })
        return self

    
    @classmethod
    def from_models(cls,
        models          : Collection[HierarchalTensorGraph],
        database_folder : Union[Path, str, FSMap, S3Path, None] = None,
        variable_depths : dict[str, Union[Int, Collection[Int]]] = {},
        datafile_kwargs : dict[Union[str, Path], dict] = {},
        verbose         : bool = False,
        find_location   : Callable | None = None,
        sources         : dict | None = None
    ) -> 'Dataset':
        """ Create a Dataset by inferring required parameters from HTGs.

        Creates list of IOSpecs from models and uses Dataset.from_specs.

        Parameters
        ----------
        models          : Collections[HierarchalTensorGraph]
            Collection of classes which inherit from HTG (or 
            their respective instantiated objects).
        database_folder : Path | str | FSMap | S3Path
            Path to the folder in which the zarr databases are stored. If a
            location defined by the model is not itself a resolvable path to
            the zarr database, the location is searched for in this folder.
        variable_depths : dict[str, Int | Collection[Int]]
            Models may sometimes allow a variable dimension size (represented
            as a None value in the shape definition) along certain axes - e.g.
            a temporal dimension can have any length when using an LSTM since 
            the model will compress all timesteps into its internal state.
            However, when reading the data from a database in that scenario,
            we need to know how many timesteps to actually read. To allow for
            this, the `variable_depths` parameter indicates the concrete window
            depth that should be used by this Dataset - thus enabling the model
            to still use Datasets which might have different sizes along the 
            variable dimension, but defining it concretely for this specific 
            Dataset object. The format is the same as `window_depths` for 
            Datafile, i.e. keys represent the dimension name, and values should
            either be a tuple of two ints representing (left size, right size);
            or a single int which represents the same value being used for both
            the left and right. For example, variable_depths={'datetime':(3,0)}
            would indicate a window with three timesteps prior to the center 
            value, and 0 timesteps after - then used by this method during 
            Dataset creation to substitute any model feature shape definitions 
            that use {'datetime': None, ...}. 
        datafile_kwargs : dict[str | Path, dict]
            Any additional keyword arguments that should be used when creating
            the Datafile objects, where the dict keys are the name of the 
            Datafile to which the respective values should be passed; e.g. 
            datafile_kwargs = {'SMAP': {'preprocessors': [backfill, coarsen]}}.
            Note the character '*' can be used as a special key to signify that
            the respective value kwargs dict should be given to all datafiles,
            e.g. {'*':{'extent': ...}} uses the given extent for all Datafiles.
            Also note there are multiple formats that will be accepted when 
            specifying the Datafile name (datafile_kwargs keys): a string that
            indicates the folder name without its extension (e.g. 
            f'{database_folder}/{name}.zarr'); the full folder name with any 
            extension included (e.g. f'{database_folder}/{name}'); the full 
            path to the database (either as a string or a Path object), in 
            which case `database_folder` parameter will not be used.
        verbose         : bool
            Whether to print information on the Datafiles being created.

        Returns
        -------
        Dataset
            A Dataset object which contains all data specified in io_specs of the given htgs.

        """

        io_specs = []
        for m in models:
            for key in ['inputs', 'outputs']:
                val = getattr(m, key)

                # Handle Node-like API
                if isinstance(val, property):
                    val = getattr(m, f'{key}_spec')
                io_specs.append(val)
                
        if not io_specs:
            raise ValueError("Model(s) do not contain inputs_spec or outputs_spec")
        
        return Dataset.from_specs( 
            io_specs, 
            database_folder,
            variable_depths,
            datafile_kwargs,
            verbose,         
            find_location,
            sources,
        )

    
    @classmethod
    def from_specs(cls, 
        io_specs        : list[IOSpec],
        database_folder : Union[Path, str, FSMap, S3Path, None] = None,
        variable_depths : dict[str, Union[Int, Collection[Int]]] = {},
        datafile_kwargs : dict[Union[str, Path], dict] = {},
        verbose         : bool = False,
        find_location   : Callable | None = None,
        sources         : dict | None = None
    ) -> 'Dataset':
        """ Create a Dataset by inferring required parameters from BaseNodes.

        IOSpec(s) contain the information necessary
        to infer Datafile locations, features, and windows. Using these and any
        other given parameters, construct and return a Dataset object.

        Parameters
        ----------
        io_specs        : list[IOSpec]
            Collection of classes which inherit from crest.base.BaseNode (or 
            their respective instantiated objects).
        database_folder : Path | str | FSMap | S3Path
            Path to the folder in which the zarr databases are stored. If a
            location defined by the model is not itself a resolvable path to
            the zarr database, the location is searched for in this folder.
        variable_depths : dict[str, Int | Collection[Int]]
            Models may sometimes allow a variable dimension size (represented
            as a None value in the shape definition) along certain axes - e.g.
            a temporal dimension can have any length when using an LSTM since 
            the model will compress all timesteps into its internal state.
            However, when reading the data from a database in that scenario,
            we need to know how many timesteps to actually read. To allow for
            this, the `variable_depths` parameter indicates the concrete window
            depth that should be used by this Dataset - thus enabling the model
            to still use Datasets which might have different sizes along the 
            variable dimension, but defining it concretely for this specific 
            Dataset object. The format is the same as `window_depths` for 
            Datafile, i.e. keys represent the dimension name, and values should
            either be a tuple of two ints representing (left size, right size);
            or a single int which represents the same value being used for both
            the left and right. For example, variable_depths={'datetime':(3,0)}
            would indicate a window with three timesteps prior to the center 
            value, and 0 timesteps after - then used by this method during 
            Dataset creation to substitute any model feature shape definitions 
            that use {'datetime': None, ...}. 
        datafile_kwargs : dict[str | Path, dict]
            Any additional keyword arguments that should be used when creating
            the Datafile objects, where the dict keys are the name of the 
            Datafile to which the respective values should be passed; e.g. 
            datafile_kwargs = {'SMAP': {'preprocessors': [backfill, coarsen]}}.
            Note the character '*' can be used as a special key to signify that
            the respective value kwargs dict should be given to all datafiles,
            e.g. {'*':{'extent': ...}} uses the given extent for all Datafiles.
            Also note there are multiple formats that will be accepted when 
            specifying the Datafile name (datafile_kwargs keys): a string that
            indicates the folder name without its extension (e.g. 
            f'{database_folder}/{name}.zarr'); the full folder name with any 
            extension included (e.g. f'{database_folder}/{name}'); the full 
            path to the database (either as a string or a Path object), in 
            which case `database_folder` parameter will not be used.
        verbose         : bool
            Whether to print information on the Datafiles being created.

        Returns
        -------
        Dataset
            A Dataset object which contains all data specified in io_specs.

        """

        # Make sure it's a list
        if not isinstance(io_specs,list):
            io_specs = [io_specs]

        universal_kwargs = datafile_kwargs.pop('*', {})
        unused_df_kwargs = unused = set(list(datafile_kwargs))

        if isinstance(database_folder, str):
            database_folder = Path(database_folder)
        elif isinstance(database_folder, FSMap):
            database_folder = S3Path(database_folder)

        def get_kwargs(source, label: str | None = None) -> dict:
            """ Get any kwargs from datafile_kwargs which matches source """
            path = Path(source) if isinstance(source, str) else source
            
            # Multiple formats are accepted when specifying the Datafile name
            if isinstance(path, (Path, S3Path)):
                options = [path, path.stem, path.name, path.as_posix()]
            else: options = [path]

            for option in [label] + options:
                if option in datafile_kwargs:
                    unused_df_kwargs.difference_update({option})
                    return universal_kwargs | datafile_kwargs[option]
            return universal_kwargs

        def get_location(io_spec, source, find):
            """ Get the zarr location from the given source, checking
                a number of ways the location could be specified. """
            if find is not None: return find(source)
            path = Path(source) if isinstance(source, str) else source

            # Any source that doesn't have 'exists' method is assumed to exist
            if not hasattr(path, 'exists'): return path

            # If the location doesn't exist, search in database_folder
            if not path.exists() and (database_folder is not None):
                path = database_folder.joinpath(path)

            # Can pass in location via datafile_kwargs
            if not path.exists():
                dfkw = get_kwargs(path) or get_kwargs(source)
                path = dfkw.get('location', path)
                path = Path(path) if isinstance(path, str) else path
                if not hasattr(path, 'exists'): return source, path

            # Also check for *folders* if extension was excluded
            if not path.exists() and isinstance(path, Path):
                glob = lambda p: p.parent.glob(f'{p.name}.*')
                dirs = lambda p: list(filter(Path.is_dir, glob(p)))
                path = ((dirs(path) or dirs(Path(source))) + [path])[0]

            assert(path.exists()), f'Unknown location for {io_spec}: {source}'
            return source, path


        def shape_key(io_spec, feature, shape) -> tuple[(str, (Int, Int))]:
            """ Create a dictionary key from the given shape, replacing None
                shape size with variable_depths value where possible, and 
                formatting each size as a two-tuple: (left, right). """
            key = []
            for k, v in sorted(shape.items(), key=lambda kv: kv[0]):
                if v is None:
                    assert(k in variable_depths),f'Spec {io_spec} needs the '+\
                        f'feature {feature} with shape {shape}, but no value'+\
                        f' was given for variable_depths to replace {k} = None'
                    v = variable_depths[k]

                # Replace total window size with depth
                v = (int(v//2),)*2 if isinstance(v, Number) else tuple(v)
                key.append((k, v))
            return tuple(key)


        def add_coord(shape_features, size, dims) -> None:
            """ Add coordinate dims with given size to location features """
            for dim in dims:
                coord_key, = key = ((dim, size),)
                
                # If the coordinate hasn't been added as a new feature set
                if key not in shape_features: 

                    # Check if any feature sets have the same dimension size
                    for shapes, feature_set in shape_features.items():
                        if coord_key in shapes:
                            feature_set.add(dim)
                            break

                    # Add the coordinate as a new shape otherwise
                    else: shape_features[key].add(dim)
                else: shape_features[key].add(dim)


        def create_datafile(source, location, window_depth, feature_set) -> Datafile:
            """ Create a Datafile object with the given parameters """
            # Remove any @ specifiers for the features
            # Format: feature@source or source>>feature
            remove_at = lambda f: f.split('@')[0].split('>>')[-1]

            # Collect all parameters and create Datafile
            df_kwargs = {
                'location'     : location,
                'features'     : sorted(map(remove_at, feature_set)),
                'window_depth' : dict(window_depth),
                'key_label'    : source
            } | get_kwargs(location, source)

            if verbose:
                print(f'\nCreating Datafile for {location.stem} with kwargs:')
                pprint(df_kwargs, compact=True, indent=4)
            return Datafile(**df_kwargs)


        # {DB name : {(window shape,) : [features, ..]}
        # {ERA5: {(None, 3, 3): ['skt', 'sp', ...]}}
        features    = dd(lambda: dd(set))
        coordinates = dd(lambda: dd(set))
        for io in io_specs:
            
            # Examine each source (zarr database) the io_spec needs
            for label, specs in io.items():
                
                # If source in sources get location
                source = (sources or {}).get(label, label)
                location = get_location(io, source, find_location)

                # Group features by their requested window shape
                for feature, shape in specs.get('coord_shapes', specs).items():
                    key = (dim,size), *_ = shape_key(io, feature, shape)

                    # Include coordinate features with other features later
                    if (len(key) == 1) and (dim == feature):
                        coordinates[location][size].add(dim)
                    else: features[location][key].add(feature)

        # Group coordinate features with other features of the same size 
        for location, coord in coordinates.items():
            [*starmap(partial(add_coord, features[location]), coord.items())]
        
        # Collate all Datafile kwargs and create the Datafile objects
        dfs = [ create_datafile(*location, window_depth, feature_set)
                 for location, size_features in features.items()
                 for window_depth, feature_set in size_features.items() ]

        # Warn if any datafile_kwargs were unused
        if len(unused): print(f'WARNING unused datafile_kwargs: {unused}')
        if verbose: print(f'Dataset with {len(dfs)} Datafiles:\n   {dfs}')
        return Dataset(dfs)

        
def uniform_chunks(
    bins : int, 
    data : list[np.ndarray], 
    size : Union[list[np.ndarray], None] = None,
):# -> list[np.ndarray]:
    """Distribute elements approximately uniformly and return bin sizes. 
    
    Notes
    -----
    The goal is to find a single set of bins over a shared coordinate axis
    (e.g. latitude), such that each coordinate vector, when binned, has an
    approximately uniform distribution of elements per bin. As computing the
    globally optimal bins requires an iterative approach, we can instead use 
    quantile averaging to compute the empirical 1D Wasserstein barycenter,
    which provides a closed-form approximation of the optimal bins. 

    Parameters
    ----------
    bins : int
        The number of bins that the data should be divided into. 
    data : list[np.ndarray]
        A list of vectors that should be used to find the bins. 
    size : Union[list[np.ndarray], None]
        The minimum chunk size that is allowed for each vector. Any chunks less
        than this size are merged with the chunk to the left or right, which is
        repeated until all chunks are at least as large as the specified size.
        Note that merging occurs globally, so that the number of chunks is 
        always the same for all vectors. Merging is disabled if size is None.

    Returns
    -------
    list[np.ndarray]
        List of vectors representing the number of elements contained in each
        bin, thus providing the chunks that ~uniformly distribute elements.

    """

    # Filter virtual dimension vectors
    real_data = lambda d: (len(d) > 1) or np.isfinite(d).all()
    available = list(filter(real_data, data))

    # Compute the empirical 1D Wasserstein barycenter
    calculate = lambda d: np.nanquantile(d, np.linspace(0, 1, bins+1))
    quantiles = np.mean(list(map(calculate, available)), axis=0)

    # Adjust endpoints to be the global min/max
    quantiles[0] = min(map(min, available))
    quantiles[-1] = max(map(max, available))

    # Set virtual dimensions to have 1 element per bin
    chunks = [np.histogram(d, quantiles)[0] if real_data(d) else np.ones(bins)
                for d in data]

    # Ensure all chunks contain at least as many elements as the window size
    if size is not None:
        w_size = np.array(size).flatten()
        chunks = list(np.array(chunks).T)
        assert(all(len(w_size) == len(c) for c in chunks)), (
            f'Incorrect size for chunks: {w_size} vs {list(map(len, chunks))}')

        # Merge bins left to right, then right to left
        for i in [0, 0]:
            while i < len(chunks):
                # If the current chunk is too small, merge with the next 
                while (w_size > chunks[i]).any() and ((i+1) < len(chunks)):
                    chunks[i] += chunks.pop(i+1)
                i += 1
            chunks = chunks[::-1]
        chunks = np.array(chunks).T
    return list(map(tuple, chunks))