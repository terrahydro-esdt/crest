from dask.diagnostics import ProgressBar
from collections.abc import Iterable, Callable
from fsspec.mapping import FSMap 
from contextlib import nullcontext, redirect_stdout
from functools import partial, cached_property
from pathlib import Path
from numbers import Number 

import cloudpickle as pkl
import numpy as np 
import dask.array as da
import dask
import io 

from crest.base.BaseSet import BaseSet
from crest.data.loading.Datafile import Datafile
from crest.data.loading.Blockset import Blockset


class Dataset(BaseSet):
    """Wraps a collection of Datafiles, thus loading multiple sources. 
    
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
    def __init__(self, locations: Iterable[Datafile|Path|str|FSMap], **kwargs):
        self.container = list(map(partial(Datafile.load, **kwargs), locations))
        assert(len(locations)), 'Must pass at least one object to init Dataset'


    @cached_property
    def dtype(self):
        """ Sample dtype """
        return np.dtype([(f'Data_{i}', df.dtype) for i,df in enumerate(self)])


    @cached_property
    def total_bytes(self) -> float:
        """ Estimate of the total input/output bytes of the data """
        inp_bytes = sum(self.data.nbytes)
        out_bytes = float(self.dtype.itemsize) * sum(map(np.prod, self.shape))
        return inp_bytes + out_bytes


    def align(self, a: str, *b):
        """ Helper used to align log text """
        return f'{a:>20}: '+' '.join(map(str, b))


    def generate_samples(self, 
        blocksize : Number = 1e8,
        compute   : bool   = True,
        verbose   : bool   = True,
        logger    : Callable = print,
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
        compute   : bool
            Whether the lazy dask array of Sample objects should be computed 
            and returned, or just a list of the `dask.delayed.Delayed` tasks 
            created for all blocks (with one block per task). If the list of 
            task objects is returned with `compute=False`, the `.compute()` 
            method can be called on each to independently compute the blocks. 
        verbose   : bool
            Whether logs should be shown when preparing and generating samples.
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


        TODO: 
            - implement test(s) for merging chunks (i.e. not enough elements in chunk for overlaps) 
            - switch Block coordinate grids to use vectors instead (though perhaps handle both?)
                - vectors by default for reduced memory usage, but may want to keep dense representation
                  handling in order to allow skewed grids 
            * diagnose changing sample number based on block count

        """
        self.logger = logger 

        # In order to avoid overlapping logs with multiple processes,
        # we accumulate all log text and log only once at the end
        if verbose: 
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

                # Rechunk the data to the requested blocksize
                self.autochunk(blocksize, verbose)

                # Create the delayed sample blocks
                samples = self.create_blocks(verbose)

        finally: 
            # Log the accumulated text prior to starting block computation
            if verbose: logger(log_txt + buffer.getvalue() + log_sep)

        # Find all samples in parallel across the created blocksets
        if compute:
            with nullcontext() if not verbose else ProgressBar():
                if verbose: logger('\nFinding all valid samples...')
                samples = da.hstack( da.compute(*samples) )
            
            if verbose: logger(f'\nFound {len(samples):,} samples')
            if save_path is not None:
                Dataset.save(samples, save_path)
        return samples 



    def autochunk(self, 
        blocksize : Number = 1e8, 
        verbose   : bool = False,
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
        verbose   : bool
            Whether logs should be shown when preparing and generating samples.

        """
        # Close (over-)estimate for the targeted number of blocks per dimension
        # (inp bytes + (estimated) out bytes) / blocksize = number of blocks
        tgt_block = int(np.ceil(self.total_bytes / blocksize))
        exp_round = lambda n: round(np.exp(n * np.ceil(np.log(tgt_block))))
        if verbose: print(self.align('Target block total', f'{tgt_block:,}'))

        # Maximum number of blocks the data could theoretically be split into, 
        # while still maintaining the required number of elements along each 
        # dimension to fulfill the requested window size
        required = lambda d_v, data, size: [np.inf if v else len(data[d])//size[d] for d,v in d_v]
        data_obj = self.data, self.window_total
        req_blks = map(required, map(zip, self.dims, self.virtual), *data_obj)
        max_blks = np.min(list(req_blks), axis=0).astype(int)
        assert(np.isfinite(max_blks).all()), f'Invalid max block size: {max_blks}'
        assert(min(max_blks) > 0), f'Requested window larger than data: {self.shape}'
        if verbose: print(self.align('Max valid blocks', max_blks))

        # Exponential rounding scheme
        # blocks = list(map(exp_round, max_blks / max_blks.sum()))

        # Get the initial state, which is the minimum between the current 
        # dataset(s) states and the maximum allowable size
        curr = self.numblocks.ix[:-1]
        init = np.min([np.lcm.reduce(curr, axis=0), max_blks], axis=0)
        seen = set()
        best = {}
        if verbose: print(self.align('Current block sizes', list(curr)))

        def optimize(state, skipdims=[]):
            """ Use a greedy approach to optimize the number of blocks per
                axis, attempting to get as close as possible to the targeted
                total number of blocks without dividing them along non-integer
                boundaries (as this would duplicate / discard far more data
                when generating blocks). Integer boundaries are maintained by 
                using only integer multiples of the original block structure. 
            """
            # Store the initial block state we're searching from
            state_tuple = best[abs(tgt_block - np.prod(state))] = tuple(state)

            # Iteratively add multiples of the original block sizes until we
            # reach the total block target, or we encounter a state seen before
            while (state_tuple not in seen) and (np.prod(state) != tgt_block):

                # If the current block combination has been seen already,
                # we don't need to try and optimize with it again
                seen.add(state_tuple)

                # The order of preference for steps that increase or decrease 
                # block size depends on the direction to the total target size
                order = 1 if np.prod(state) < tgt_block else -1

                # Each axis can step up or down, but must fulfill the condition
                # 0 < new size < max. If the preferred step direction fails, we
                # can instead use the alternative direction choice as backup
                st1,st2 = [init * inc_dec for inc_dec in [1,-1][::order]]
                isvalid = lambda new_state: (0<new_state)&(new_state<=max_blks)
                backups = np.where(isvalid(state + st2), st2, np.nan)
                stepdir = np.where(isvalid(state + st1), st1, backups)

                # Helpers:
                # - keep the step choice if finite and not part of skipdims
                # - swap a given index in the state with a new value
                # - calculate the difference between target and the new state 
                keep = lambda i,s: (i not in skipdims) and np.isfinite(s)
                swap = lambda i,s: [v + s*(j==i) for j,v in enumerate(state)]
                diff = lambda i,s: (i, abs(tgt_block - np.prod(swap(i, s))))

                # Calculate distance to the target for each valid state option
                options = [diff(*s) for s in enumerate(stepdir) if keep(*s)]
                if len(options) == 0: break

                # Select the state which minimizes distance to the target
                idx, option = min(options, key=lambda i_o: i_o[1])
                state[idx] += stepdir[idx]
                state_tuple = best[option] = tuple(state)

            seen.add(state_tuple)

        # First optimize using the current blocks as the initial state
        optimize(init.copy())

        # Then if we're increasing the number of blocks, optimize with a new
        # state for each axis in the data (increasing the blocks to the max
        # targeted/allowed size for each respective axis)
        if np.prod(init) < tgt_block:
            for i, c in enumerate(init):
                targeted = tgt_block // np.prod(init)
                allowed  = max_blks[i] // c
                incstate = init.copy()
                incstate[i] *= max(1, min(targeted, allowed))
                optimize(incstate)

        # Otherwise the current number of blocks needs to be decreased along
        # one or more dimensions (which is done by evenly combining existing
        # blocks using their prime factors)
        # The way this is currently set up will artificially limit the lower
        # end of block sizes, as only 2,3,5 are used as (single) divisors.
        # Eventually we need to remove this limit to allow more control over
        # blocks via blocksize.
        else: 
            for dim in range(len(init)):
                primefac = [[f for f in [2, 3, 5] if c%f == 0] for c in init]
                skipdims = set()
                decstate = init.copy() 

                # Reduce the number of blocks as much as necessary, iterating
                # repeatedly over axes until either the number of blocks fall
                # below the target or we run out of prime factors
                while np.prod(decstate) > tgt_block:
                    for factor in primefac[dim]:
                        if (decstate[dim] % factor) == 0:
                            decstate[dim] = init[dim] // primefac[dim].pop(0)
                            skipdims.add(dim)
                            break    
                        else: primefac[dim].pop(0)

                    dim = (dim + 1) % len(decstate)
                    if max(map(len, primefac)) == 0: break

                # Any axes which are reduced need to be skipped over when 
                # optimizing, as it would otherwise allow non-integer multiples
                # with respect to the original block sizes
                optimize(decstate, skipdims)

        # Bin the data into histograms with the specified number of blocks
        # in order to get the final chunk sizes
        blocks = best[min(best)]
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



    def create_blocks(self, verbose: bool = False):
        # Get the current number of blocks
        chunks  = self.chunks.ix[:-1]
        current = self.numblocks.ix[:-1]
        blocks  = np.array(current[0])
        assert(all(((current == blocks) | self.virtual).map(all))), current

        # Calculate required overlaps for each Datafile
        max_res = np.nanmax(self.max_resolution, axis=0)
        idx_res = np.nanargmax(self.max_resolution, axis=0)
        skipdim = idx_res == np.arange(len(self))[:, None]
        overlap = self.calculate_overlap(max_res, _map=[skipdim])
        if verbose: 
            lbl = lambda dims, ov: str({dims[k]: v for k,v in ov.items()})
            ind = '\n                '
            txt = ind + ind.join(map(lbl, self.dims, overlap))
            print(self.align('Overlaps', txt))

        # Apply the required overlaps to each Datafile
        prepped = self.apply_overlap(_map=[overlap])
        blksize = (self.total_bytes / np.prod(blocks)) / 1e6
        element = sum(map(np.prod, [list(map(np.mean, c)) for c in chunks]))

        if verbose: 
            print('\nResults:\n  ' + '\n  '.join([
                f'Split into {np.prod(blocks)} total block(s)',
                f'~{blksize:,.1f} MB/block'.replace('.0',''),
                f'~{element:,.0f} items/block'
            ]))
        # return [(lambda: dask.delayed(Blockset(p, logger=getattr(self, 'logger', print)).find_matches)()) 
        #         for p in zip(*prepped, strict=True)]

        # Create a list of Blocksets, where each Blockset 
        # contains exactly one Block from each Datafile
        blocksets = map(dask.delayed(Blockset), zip(*prepped, strict=True))
        matches   = map(lambda bset: bset.find_matches(), blocksets)
        return list(matches)



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
