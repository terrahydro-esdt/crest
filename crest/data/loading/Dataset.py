from collections.abc import Iterable, Iterator, Callable, Collection
from dask.diagnostics import ProgressBar
from dask.delayed import Delayed
from fsspec.mapping import FSMap 
from contextlib import nullcontext, redirect_stdout
from functools import partial, cached_property
from pathlib import Path
from numbers import Number 
from logging import Logger 
from typing import Union 

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

from crest.utils import Stopwatch, optimize_blocks
from crest.base import BaseSet
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
    def __init__(self, locations: Iterable[Union[Datafile, Path, str, FSMap]], **kwargs):
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


    def align(self, a: str, *b):
        """ Helper used to align log text """
        return f'{a:>20}: '+' '.join(map(str, b))


    def generate_samples(self, 
        blocksize : Number = 1e9,
        numblocks : int | Collection[int] = 0,
        compute   : bool   = True,
        verbose   : bool   = True,
        optimize  : bool   = True,
        shuffle   : bool   = False,
        logger    : Logger | None = None,
        save_path : str | Path | None = None,
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



    def create_blocks(self, 
        blocksize : Number = 1e8,
        numblocks : int | Collection[int] | None = 0,
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

        # Either block the data to an exact number per dimension (e.g. [1,5,9])
        # or try to automatically find a reasonable common blocking scheme 
        # based on the scheme currently used across all Datafiles
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
            lbl = lambda dims, ov: str({dims[k]: v for k,v in ov.items()})
            ind = '\n                '
            txt = ind + ind.join(map(lbl, self.dims, overlap))
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
        numblocks : int    = 0,
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
        # Close (over-)estimate for the targeted number of blocks per dimension
        # (inp bytes + (estimated) out bytes) / blocksize = number of blocks
        if numblocks>0: tgt_blks = numblocks 
        else:           tgt_blks = int(np.ceil(self.total_bytes / blocksize))
        if verbose: print(self.align('Target block total', f'{tgt_blks:,}'))

        # Maximum number of blocks the data could theoretically be split into, 
        # while still maintaining the required number of elements along each 
        # dimension to fulfill the requested window size
        required = lambda d_v, data, size: [np.inf if v else len(data[d])//size[d] for d,v in d_v]
        data_obj = self.data, self.window_total
        req_blks = map(required, map(zip, self.dims, self.virtual), *data_obj)
        max_blks = np.min(list(req_blks), axis=0).astype(int)
        cur_blks = self.numblocks.ix[:-1]
        assert(np.isfinite(max_blks).all()), f'Invalid max block size: {max_blks}'
        assert(min(max_blks) > 0), f'Requested window larger than data: {self.shape}'

        if verbose: 
            print(self.align('Max valid blocks', max_blks))
            print(self.align('Current block sizes', list(cur_blks)))

        # Bin the data into histograms with the specified number of blocks
        # in order to get the final chunk sizes
        blocks = optimize_blocks(cur_blks, tgt_blks, max_blks)
        virtual= lambda x: (x.values.size == 1) and np.isnan(x.values[0])
        rm_nan = lambda blk, dat: list(range(blk)) if virtual(dat) else dat
        binner = lambda blk, dat: np.histogram(rm_nan(blk, dat), int(blk))[0]
        mapper = lambda dim, dat: map(binner, blocks, [dat[d] for d in dim])
        chunks = list(map(list, map(mapper, self.dims, self.data)))
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
    def save(cls, samples: da.Array, filename: str | Path):
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
    def load(cls, filename: str | Path) -> da.Array:
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
        cache_dir : Path | str  = '.',
    ):
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
        cache_dir : Path | str
            Location for the cached data to be stored. By default, data is 
            cached in `./Cache`. 

        """
        with Stopwatch(f'Cached {len(self)} Datafiles'):
            # Rechunk the data first
            self.generate_samples(**{
                'numblocks' : numblocks, 
                'compute'   : False, 
                'verbose'   : False,
                'optimize'  : False,
            })
            self._cache(overwrite, cache_dir, _delay=False)