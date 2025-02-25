from collections.abc import Iterable, Iterator, Callable, Collection
from collections import defaultdict as dd
from dask.diagnostics import ProgressBar
from dask.delayed import Delayed
from fsspec.mapping import FSMap 
from contextlib import nullcontext, redirect_stdout
from functools import partial, cached_property
from itertools import starmap
from pathlib import Path
from numbers import Number, Integral as Int 
from logging import Logger 
from typing import Union 
from pprint import pprint

import cloudpickle as pkl
import xarray as xr
import numpy as np 
import dask.array as da
import dask
import traceback
import logging
import shutil
import zarr
import io 

from crest.utils import Stopwatch, optimize_blocks, S3Path
from crest.base import BaseSet, BaseNode
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


    def align(self, a: str, *b, n=20):
        """ Helper used to align log text """
        return f'{a:>{n}}: '+' '.join(map(str, b))


    def generate_samples(self, 
        blocksize : Number = 1e9,
        numblocks : Union[int, Collection[int]] = 0,
        compute   : bool   = True,
        verbose   : bool   = True,
        optimize  : bool   = True,
        shuffle   : bool   = False,
        logger    : Union[Logger, None] = None,
        loglevel  : Union[int, None] = None,
        save_path : Union[str, Path, None] = None,
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
        blocksize : Number
            Number of bytes that should be allocated to each block and worked
            on in parallel (default=1e8; 100MB). Note that this is just a proxy
            for the amount memory that will be used when computing a block, as 
            the actual amount used is dependent upon the number of matches that
            are found in the block and thus can vary significantly. 
        numblocks : int 
            Alternative to giving a blocksize value. If numblocks > 0, the 
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
        loglevel  : int | None
            Log level that should be used by the logger, e.g. logging.INFO. If
            None is given, the default level set by the logger is used.  
        save_path : str | Path | None
            Save the generated sample array to a pickle file at the given path.

        Returns
        -------
        dask.array.Array | list[dask.delayed.Delayed]
            Either the lazy array which contains all valid samples which were 
            found (if `compute=True`, the default); or a list of the Delayed
            task objects that correspond to the individual block computations.

        """
        if logger is not None:
            self.__dict__['logger'] = logger
        if loglevel is not None:
            self.logger.setLevel(loglevel)
            
        # In order to avoid overlapping logs with multiple processes,
        # we accumulate all log text and log only once at the end
        log_sep = ''.join(['_']*60) + '\n'
        dat_sep = ''.join(['-']*60)
        log_txt = f'\n{log_sep}\nCurrent data:\n{dat_sep}'
        for c in self.container:
            log_txt += f'\n\n{c.data}'
        log_txt += f'\n\n{dat_sep}\n\nPreparing data...\n'

        # Capture stdout to log appropriately
        try:     
            with redirect_stdout(io.StringIO()) as buffer:
                # Ensure all Datafiles are aware of all dimensions
                # i.e. create virtual dimensions where necessary
                self.ensure_dims( set.union(*map(set, self.dims)) )

                # Ensure the coordinate extents are aligned across Datafiles
                # i.e. so that blocks will refer to the same coordinate areas
                # Unfortunately, still not working properly / requires large
                # amount of memory for certain datasets
                # if len(self) > 1: self.ensure_extents()

                # Create the delayed sample blocks
                samples = self.create_blocks(blocksize, numblocks, verbose, optimize, shuffle)
        except: 
            verbose = True
            raise

        finally: 
            # Log the accumulated text prior to starting block computation
            if verbose: self.logger.info(log_txt + buffer.getvalue() + log_sep)

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
        blocksize : Number = 1e8,
        numblocks : Union[int, Collection[int], None] = 0,
        verbose   : bool = False,
        optimize  : bool = True,
        shuffle   : bool = False,
    ) -> list:
        """Block data into the requested configuration.

        Parameters
        ----------
        blocksize : Number
            Number of bytes that should be allocated to each block and worked
            on in parallel (default=1e8; 100MB). Note that this is just a proxy
            for the amount memory that will be used when computing a block, as 
            the actual amount used is dependent upon the number of matches that
            are found in the block and thus can vary significantly. 
        numblocks : int 
            Alternative to giving a blocksize value. If numblocks > 0, the 
            requested number of blocks is the target block total. While the 
            exact number of blocks is not always possible to create, an attempt
            is made to get as close as possible to the requested value.
        verbose   : bool
            Whether logs should be shown when preparing and generating samples.
        optimize  : bool
            Whether dask.optimize should be used on the full task graph. This
            can speed up access when batching data, but incurs a higher cost
            when initially generating sample blocks. 
        shuffle   : bool
            To the extent possible, shuffle sample ordering.
        
        Returns
        -------
        list
            List of delayed Blockset.find_matches functions.

        """
        # Attempt to automatically block the data based on total block count
        if isinstance(numblocks, int):
            self.autochunk(blocksize, numblocks, verbose)
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
                    blocks[indices[i]] = max_block(shapes, 10)

            if verbose:
                print(f'\tcurrent blocks: {self.numblocks}')
                print(f'\t target blocks: {blocks}')

            # Update Datafile chunks to create the required number of blocks
            chunks = self.update_blocks(blocks)
            if verbose: 
                if any(chunks): print(f'\trechunked with: {chunks}')
                print(f'\t result blocks: {self.numblocks[0][:-1]}')

        # Validate all chunks are at least as large as requested window size
        max_blocks = np.min(list(self.max_valid_blocks), axis=0).astype(int)
        if any(blk > maxim for blk, maxim in zip(blocks, max_blocks)):
            raise Exception(f'Specified blocks {blocks} are greater than the '+
                    f'maximum valid number of blocks {max_blocks}')

        # This can only be run when not caching, since zarr doesn't allow us to 
        # write non-uniform chunks. Matching up anisotropic grids between Datafiles
        # requires us to have non-uniformly sized chunks however, and so this must
        # be run by any object which is generating matchups, e.g. Batcher
        if optimize: self.autochunk(numblocks=blocks)

        if verbose: 
            chunks = lambda df: [f'{d:>10} = {c}' for d,c in zip(df.dims, df.chunks)]
            to_str = lambda df: f'{df}'+'\n\t\t  '.join(['']+chunks(df))
            print(self.align('Chunks', '\n\t\t'+'\n\t\t'.join(map(to_str, self))))

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
        overlap = self.calculate_overlap(max_res, dimension_blks=blocks, _map=[skipdim])

        if verbose: 
            buf = max(map(len, map(str, self))) + 1
            lbl = lambda df, dim, ov: f'{str(df):>{buf}} = '+'   '.join(f'{dim[k]}: {v}' for k,v in ov.items())
            ind = '\n                '
            txt = ind + ind.join(map(lbl, self, self.dims, overlap))
            print(self.align('Overlaps', txt))

            nbytes = (self.total_bytes / np.prod(blocks)) / 1e6
            nelems = sum(map(np.prod, [list(map(np.mean, c)) for c in chunks]))
            print('\nResults:\n  ' + '\n  '.join([
                f'Split into {np.prod(blocks)} total block(s)',
                f'~{nbytes:,.1f} MB/block'.replace('.0',''),
                f'~{nelems:,.0f} items/block'
            ]))

        # Create a list of Blocksets, where each Blockset 
        # contains exactly one Block from each Datafile
        prepped    = self.apply_overlap(optimize=optimize, _map=[overlap])
        create_set = dask.delayed(partial(Blockset, logger=self.logger, shuffle=shuffle)) 
        blocksets  = map(create_set, zip(*prepped, strict=True))
        matches    = map(lambda bset: bset.find_matches, blocksets)
        return list(matches)



    def autochunk(self, 
        blocksize : Number = 1e8, 
        numblocks : Union[int, Collection[int]] = 0,
        verbose   : bool   = False,
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
        blocksize : Number
            Number of bytes that should be allocated to each block and worked
            on in parallel (default=1e8; 100MB). Note that this is just a proxy
            for the amount memory that will be used when computing a block, as 
            the actual amount used is dependent upon the number of matches that
            are found in the block and thus can vary significantly. 
        numblocks : int 
            Alternative to giving a blocksize value. If numblocks > 0, the 
            requested number of blocks is the target block total. While the 
            exact number of blocks is not always possible to create, an attempt
            is made to get as close as possible to the requested value.
        verbose   : bool
            Whether logs should be shown when preparing and generating samples.

        """
        if isinstance(numblocks, int):
            # Close (over-)estimate for the targeted number of blocks per dimension
            # (inp bytes + (estimated) out bytes) / blocksize = number of blocks
            if numblocks>0: tgt_blks = numblocks 
            else:           tgt_blks = int(np.ceil(self.total_bytes / blocksize))
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

            if verbose: 
                print(self.align('Max valid blocks', max_blks))
                print(self.align('Current block sizes', list(cur_blks)))

            # Bin the data into histograms with the specified number of blocks
            # in order to get the final chunk sizes
            blocks = optimize_blocks(cur_blks, tgt_blks, max_blks)
        else: blocks = numblocks
        if verbose: print(self.align('Pre-merged blocks', blocks))
        
        # First get the average bin edge locations for all dimensions
        virtual= lambda x: (x.values.size == 1) and np.isnan(x.values[0])
        rm_nan = lambda blk, dat: list(range(blk)) if virtual(dat) else dat
        binner = lambda blk, dat: np.histogram(rm_nan(blk, dat), int(blk))
        mapper = lambda dim, dat: zip(*map(binner, blocks, [dat[d] for d in dim]))
        _,edges = zip(*map(list, map(mapper, self.dims, self._typed_data)))

        # Extend first and last edges to be the min/max values
        average = []
        for i, edge in enumerate(zip(*edges)):
            edge = [e for e, dims, dat in zip(edge, self.dims, self._typed_data, strict=True) 
                    if not virtual(dat[dims[i]])]
            minim, avg, maxim = np.min(edge,0), np.mean(edge,0), np.max(edge,0) 
            avg[0] = np.min(minim)
            avg[-1] = np.max(maxim)
            average.append(avg)

        # Then apply the averaged edges to get chunk sizes for each datafile
        binner2 = lambda blk, dat, edges: (np.ones(blk),edges) if virtual(dat) else np.histogram(dat, edges)
        mapper2 = lambda dim, dat: zip(*map(binner2, blocks, [dat[d] for d in dim], average))
        chunks,_ = zip(*map(list, map(mapper2, self.dims, self._typed_data)))
        assert(len(set(map(len, chunks))) == 1), chunks

        def merge(chunks, w_size):
            """
            Histogram can create bins containing fewer than `window depth` 
            items. So, after histogram bin calculation, need to post-process
            in order to merge bins with fewer than the necessary elements.
            Those merges need to be mirrored across grids, however - such that
            respective bin items end up in the same block across all grids.
            """
            assert(len(set(map(len, chunks))) == 1), chunks 
            assert(len(w_size) == len(chunks)), [chunks, w_size]
            w_size = np.array(w_size)
            chunks = list(np.array(chunks).T)

            # Merge left to right, then right to left
            for _ in range(2):
                i = 0
                while i < len(chunks):
                    while (w_size > chunks[i]).any() and ((i+1) < len(chunks)):
                        chunks[i] += chunks.pop(i+1)
                    i += 1
                chunks = chunks[::-1]
            return map(tuple, np.array(chunks).T)

        w_size = self.window_total.ix[self.dims]
        chunks = list(zip(*map(merge, zip(*chunks), zip(*w_size))))
        blocks = [len(c) for c in chunks[0]]
        assert(all(len(set(map(len, c))) == 1 for c in zip(*chunks))), chunks
        if verbose: print(self.align('Created blocks', blocks))
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
        numblocks : Collection[int], 
        overwrite : bool = False, 
        cache_dir : Union[Path, str, FSMap, S3Path] = 'Cache',
        verbose   : bool = False,
        fastcheck : bool = False,
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
        fastcheck : bool
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

        cpath = lambda df: cache_dir.joinpath(df.name, f'{df.config_hash}.zarr')
        paths = list(map(cpath, self))
        if not (fastcheck and all(p.exists() for p in paths)):
            with Stopwatch(f'Cached {len(self)} Datafiles at {cache_dir}'):
                # Rechunk the data first
                self.generate_samples(**{
                    'numblocks' : numblocks, 
                    'compute'   : False, 
                    'verbose'   : verbose,
                    'optimize'  : False,
                })
                self._cache(overwrite, verbose, cache_dir, _delay=False)
            print()

        # Reinitialize Datafiles with their cached data
        # Anything handled by the cache (e.g. extent) can be dropped
        else: 
            if verbose: print(f'All caches already exist ({fastcheck=})')
            self.reset(extent={},preprocessors=[],_kwmap={'location': paths})
        return self



    @classmethod
    def from_models(cls, 
        models          : Collection[BaseNode],
        database_folder : Union[Path, str, FSMap, S3Path, None] = None,
        variable_depths : dict[str, Union[Int, Collection[Int]]] = {},
        datafile_kwargs : dict[Union[str, Path], dict] = {},
        verbose         : bool = False,
    ) -> 'Dataset':
        """ Create a Dataset by inferring required parameters from BaseNodes.

        BaseNode models (classes or objects) contain the information necessary
        to infer Datafile locations, features, and windows. Using these and any
        other given parameters, construct and return a Dataset object.

        Parameters
        ----------
        models          : Collection[BaseNode]
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
            A Dataset object which contains all data necessary for the models.

        """
        universal_kwargs = datafile_kwargs.pop('*', {})
        unused_df_kwargs = unused = set(list(datafile_kwargs))

        if isinstance(database_folder, str):
            database_folder = Path(database_folder)
        elif isinstance(database_folder, FSMap):
            database_folder = S3Path(database_folder)

        def get_kwargs(source) -> dict:
            """ Get any kwargs from datafile_kwargs which matches source """
            path = Path(source) if isinstance(source, str) else source
            
            # Multiple formats are accepted when specifying the Datafile name
            if isinstance(path, (Path, S3Path)):
                options = [path, path.stem, path.name, path.as_posix()]
            else: options = [path]

            for option in options:
                if option in datafile_kwargs:
                    unused_df_kwargs.difference_update({option})
                    return universal_kwargs | datafile_kwargs[option]
            return universal_kwargs


        def get_location(model, source):
            """ Get the zarr location from the given source, checking
                a number of ways the location could be specified. """
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
                if not hasattr(path, 'exists'): return path

            # Also check for *folders* if extension was excluded
            if not path.exists() and isinstance(path, Path):
                glob = lambda p: p.parent.glob(f'{p.name}.*')
                dirs = lambda p: list(filter(Path.is_dir, glob(p)))
                path = ((dirs(path) or dirs(Path(source))) + [path])[0]

            assert(path.exists()), f'Unknown location for {model}: {source}'
            return path


        def shape_key(model, feature, shape) -> tuple[(str, (Int, Int))]:
            """ Create a dictionary key from the given shape, replacing None
                shape size with variable_depths value where possible, and 
                formatting each size as a two-tuple: (left, right). """
            key = []
            for k, v in sorted(shape.items(), key=lambda kv: kv[0]):
                if v is None:
                    assert(k in variable_depths), f'Model {model} needs the '+\
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


        def create_datafile(location, window_depth, feature_set) -> Datafile:
            """ Create a Datafile object with the given parameters """
            # Remove any @ specifiers for the features
            remove_at = lambda f: f.split('@')[0]

            # Collect all parameters and create Datafile
            df_kwargs = {
                'location'     : location,
                'features'     : sorted(map(remove_at, feature_set)),
                'window_depth' : dict(window_depth),
            } | get_kwargs(location)

            if verbose:
                print(f'\nCreating Datafile for {location.stem} with kwargs:')
                pprint(df_kwargs, compact=True, indent=4)
            return Datafile(**df_kwargs)


        # {DB name : {(window shape,) : [features, ..]}
        # {ERA5: {(None, 3, 3): ['skt', 'sp', ...]}}
        features    = dd(lambda: dd(set))
        coordinates = dd(lambda: dd(set))
        for model in models:

            # Check input and output shape definitions for the model
            for io in [model.inputs, model.outputs]:

                # Examine each source (zarr database) the model needs
                for source, feature_shapes in io.items():
                    location = get_location(model, source)

                    # Group features by their requested window shape
                    for feature, shape in feature_shapes.items():
                        key = (dim,size), *_ = shape_key(model, feature, shape)

                        # Include coordinate features with other features later
                        if (len(key) == 1) and (dim == feature):
                            coordinates[location][size].add(dim)
                        else: features[location][key].add(feature)

        # Group coordinate features with other features of the same size 
        for location, coord in coordinates.items():
            [*starmap(partial(add_coord, features[location]), coord.items())]

        # Collate all Datafile kwargs and create the Datafile objects
        dfs = [ create_datafile(location, window_depth, feature_set)
                 for location, size_features in features.items()
                 for window_depth, feature_set in size_features.items() ]
    
        # Warn if any datafile_kwargs were unused
        if len(unused): print(f'WARNING unused datafile_kwargs: {unused}')
        if verbose: print(f'Dataset with {len(dfs)} Datafiles:\n   {dfs}')
        return Dataset(dfs)