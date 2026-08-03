from __future__ import annotations
from collections.abc import Collection
from collections import defaultdict
from fsspec.mapping import FSMap 
from tlz.curried import valfilter, merge
from contextlib import nullcontext, contextmanager
from functools import partial, cached_property
from itertools import chain, islice, zip_longest
from tqdm.auto import tqdm
from pathlib import Path
from ctypes import c_int, c_bool
from typing import Union, ContextManager, Callable
from queue import Empty, Full, Queue
from math import ceil

import multiprocessing.sharedctypes
import multiprocessing.synchronize
import multiprocessing.queues
import multiprocessing as mp
import dask.array as da
import pickle as pkl
import numpy as np
import faulthandler
import traceback
import threading
import logging
import numbers
import signal
import psutil
import shutil
import time
import sys
import os
import re
import gc

from crest.utils import Stopwatch, TimedHandler, S3Path, ensure_with, limit_calls
from .ThreadedFunction import ThreadedFunction
from ..loading import Dataset, StructuredDataset, SampleSet, Block

logger = logging.getLogger(__name__)

# Some type checkers report Literal numbers aren't compatible with Number
Number = numbers.Real | int | float

# Amount of time to sleep when waiting on something to complete
#   This should be a small value, but large enough that there won't be
#   a large number of GIL releases which can cause additional slowdown
WAIT_TIME = 0.01

def identity(batcher, x):
    return x

    
class Batcher:
    """Handles creating batches of data samples.

    Parameters
    ----------
    dataset    : Dataset | StructuredDataset
        Crest Dataset or StructuredDataset object.
    batch_size : int
        Number of samples that each batch should contain.
    epoch_size : int | None
        Number of batches that each epoch should contain. See `block_size`.
    block_size : int | None
        Number of samples that each block should contain. Both `epoch_size` and
        `block_size` control the same logic, and so only one should be used at 
        a time. `block_size` determines how many samples should be pulled from 
        each block of the data (at most, since a block may not have enough 
        samples due to NaN values); `epoch_size` indirectly controls the number
        of samples which are pulled from each block by computing::

            block_size = (epoch_size*batch_size) / n_blocks

        In this way, both parameters affect how many batches are required to 
        complete a full cycle through the spatiotemporal extent of the data 
        (i.e. an epoch); epoch_size takes care of the sample calculation
        automatically, whereas block_size allows more fine-grained control. 
        In addition, setting both parameters to None (the default) will return 
        all samples across all blocks in an epoch - and so care must be taken
        to ensure a full epoch is completed if the entire data extent needs to
        be sampled, as returning all samples can mean a very large number of
        batches are required to actually complete a full epoch. 
    features   : list
        List of features that should be extracted from batch Samples. By
        default, no features are extracted, and a batch will be a list of
        Sample objects. When given, a batch will have the same nested layout
        as the feature list, and contain dictionaries with features as keys
        and numpy arrays (shaped [batch_size, ...]) as values. For example,
        features=[['a', 'b'], ['c']] would result in batches that look like:

        - [{'a':<batch of 'a' values>, 'b': <batch of 'b' values>},
          {'c': <batch of 'c' values>}]

        If any features are missing in the dataset, an exception is raised. 
        If an empty list is given, it is equivalent to selecting all available
        features. Note that any nested list in features must contain 
        homogeneous types; i.e. all strings, or all lists. Examples:

        - [['a'], ['b', 'c'], [['d', 'e'], ['f']]] is valid
        - [['a'], 'b'] is not valid
        - [['a'], ['b', ['c', 'd']]] is not valid


    workers    : int
        Number of processes to use to create batches in parallel. Note that
        workers <= 0 means that only threads will be used to generate batches;
        this configuration is about 10% faster than using workers=1, but can
        cause difficulties with closing the threads since threads cannot be
        terminated independently of the main thread in python.
    threads    : int
        Number of threads to use to create batches in parallel. Note that twice
        this number of threads are actually used, as one set is used for
        loading dataset blocks and the other set is used for loading samples
        from the chunks within a block. Must be >= 1.
    shuffle    : bool
        Whether samples should be shuffled to generate batches, or returned in
        the original order of the sample array. If correctly ordered batches
        are desired with shuffle=False, multiple processes and threads should 
        not be used as completion progress cannot be consistently ordered.
    repeat     : bool
        Whether the Batcher should repeatedly iterate over the blocks once all
        have been used to yield samples (i.e. an epoch has been completed). 
        Note using True means the Batcher will yield batches indefinitely.
    duplicate  : bool
        Whether batch samples across processes can be duplicated. With this set
        to True, each sample can be encountered `worker` times in an epoch,
        albeit at different times due to shuffling. This is particuarly useful
        when the number of blocks for a dataset is fewer than the number of
        workers that can be active - as without duplication, the number of
        workers is limited to the number of blocks in the dataset (one block
        per worker).
    continuous : bool
        If True, dataset sampling is continuous rather than pausing once the
        queue is filled with samples. In other words, the batcher will simply
        discard blocks of samples if the queue is full, rather than waiting 
        for room in the queue to add those samples. This allows more uniformly
        random sampling of the overall data space when there are far more 
        samples than the desired number of batches to be generated. Note that
        repeat must be also be True for this setting to be used. 
    block_bytes  : Number
        Number of bytes that should be allocated to each block and worked
        on in parallel (default=1e8; 100MB). Note that this is just a proxy
        for the amount memory that will be used when computing a block, as 
        the actual amount used is dependent upon the number of matches that
        are found in the block and thus can vary significantly. 
    max_queue  : int
        Controls the maximum number of batch _groups_ that can be held by the
        queue which communicates with background processes. Note that workers
        continue generating batches beyond this amount; they will just block
        when attempting to add further batches to the queue once it is full.
        When `workers` < 1 this parameter has no effect. 
    task_bytes : float
        Maximum number of bytes that should comprise a _batcher task. This
        value controls the tradeoff between memory usage and batch throughput,
        with a higher number of bytes per task leading to more batches created
        in a single task. Having a value too large can cause long periods
        without batches however, even if the average throughput is marginally
        greater. In general, larger individual samples (those with large
        windows in multiple dimensions, for instance) will require larger
        bytes per task in order to achieve optimal throughput.
    log_file    : bool
        File that logs should be written to.
    log_level   : int
        Level that log file will display. Should be a level defined by the
        logging module, i.e. logging.INFO, logging.DEBUG, etc.
    log_delay   : float
        When multiple workers are used, `log_delay` controls the amount of time
        (in seconds) that will be used as a buffer for logging records. In this
        way, logs coming from a given worker will be grouped together when they
        are near in time to one another. This has the consequence of creating
        log files in which the records are not strictly ordered in time, as
        records may be grouped such that writing them to the log file is 
        delayed by <= `log_delay` sec. To disable this functionality, set 
        log_delay=0 (or any value <= 0).
    block_sync : bool
        Controls block synchronization across workers: once a worker finishes
        computing samples for a block, it waits for all other workers to finish
        a block before moving on to a new block. This is made available to be 
        used in combination with `valid_percents` or `drop_datafiles`, when it
        is useful to have a guarantee for batches to be generated from all 
        configurations before repeating a configuration. It is recommended for 
        this to be set to False since it will force workers to wait for other 
        workers before moving on with their workload (thus slowing overall
        batch generation to be proportional to the slowest workload).
    epoch_sync : bool
        If True, workers will wait once they complete an epoch for all other
        workers to complete it. Once all workers have finished the current
        epoch, they continue on to the next epoch. 
    numblocks  : int | None
        Alternative to giving a block_bytes value. If numblocks > 0, the 
        requested number of blocks is the target block total. While the 
        exact number of blocks is not always possible to create, an attempt
        is made to get as close as possible to the requested value. If None
        is given, the block structure of the dataset is used.
    prefetch   : int | bool
        Controls prefetching batches in a separate queue that is managed by a
        background thread.  If False, prefetching is disabled and batches are 
        streamed directly from worker processes. If True (default), prefetching
        is enabled and the prefetch queue uses the `max_queue` parameter as its
        maximum size. An integer may also be given, which enables prefetching
        and sets the maximum size of the prefetch queue to that value. 
    prefetcher : ContextManager | Callable
        A function or context manager (factory) which is applied to batches
        when they are prefetched. This allows additional work to be performed
        in a background thread, if any post-processing needs to be applied to
        batches. If a context manager is passed, it should yield the actual 
        prefetcher function; it will be entered just before batches begin
        to be yielded, and exited after all batches have been yielded. Note
        that using `prequeuer` should be preferred, unless the function uses
        objects which cannot be pickled.
    prequeuer  : ContextManager | Callable
        Same as `prefetcher`, but the function is applied prior to adding
        batches to the multiprocessing worker queue - meaning it is applied
        by workers in parallel, and thus can improve speed considerably if
        the batch post-processing function requires significant work. Note
        that the function must be pickle-able since it is distributed to the
        worker processes; if for some reason it cannot be pickled, use the
        `prefetcher` parameter instead. 
    seed       : int | None
        Seed for reproducible randomness.
    cache_dir  : Union[Path, str, FSMap, S3Path, None]
        Location where data blocks should be cached. Note that the caching
        performed by the Dataset class is distinct from the caching performed
        here: by caching the Dataset in the exact format required for creating
        batches, we minimize the number of file reads (e.g. from overlapping 
        block regions) as well as chunk the zarr data asymmetrically by padding
        non-symmetric chunks (i.e. optimizing Datafile chunking). If `None` is
        given (default), the data is not cached and blocks are generated from
        the Datafile locations directly. 
    overwrite  : bool
        Whether data should be re-cached if it already exists in `cache_dir`.
    valid_percents : list[dict[str, dict]]
        Note: in general, the `drop_datafiles` parameter should be preferred 
        over the use of `valid_percents`. With multiple sources of data, it may
        be necessary to allow for some sources to be missing to find any valid 
        samples from others. For instance, sparse data coming from ground 
        stations (e.g. FLUXNET) would have few (if any) spatial overlaps with 
        other sparse data sources (e.g. SNOTEL); in order to generate samples 
        from both of these sources, we can rotate through them by allowing and
        disallowing missing values for each of them with this parameter::

            valid_percents=[ {'FLUXNET' : {'datetime':0}},
                             {'SNOTEL'  : {'datetime':0}}, ]

        This has two valid_percent configurations to rotate through, which will
        be used independently by each worker. In other words, when a worker is
        preparing to compute samples for a block, it will first randomly choose
        one of the configurations provided in the `valid_percents` list, then
        use that chosen configuration to specify the appropriate Datafile 
        `valid_percent` values for that block computation. For detail on the
        format of `valid_percent`, see the docstring for Datafile. 
    drop_datafiles : list[list[str]]
        Similar to the `valid_percents` parameter, this keyword addresses the
        same goal of allowing a single Batcher to generate samples from
        multiple different Datafile combinations. However, `drop_datafiles` 
        will actually drop the specified Datafiles during block computation
        rather than just allowing values to be missing. Consequently, computing
        blocks is much faster (as there are fewer Datafiles to match up) - but 
        Datafiles which are dropped in a block will produce NaN features, even
        if those features are actually valid for a given sample location. The
        usage of this parameter follows the same format of `valid_percents`,
        but only requires the name of Datafiles to drop::

            drop_datafiles = [['FLUXNET'], ['SNOTEL']]

        This list contains two configurations that will be rotated through: 
        one which drops the FLUXNET Datafile during block computation, and one
        which drops the SNOTEL Datafile. Multiple names can be specified for a
        single configuration, and any number of configurations can be provided.
        For both of these parameters, the integer index of the Datafile within
        the Dataset can also be used instead of the Datafile label, in the case
        where Datafile labels aren't meaningful (though care must be taken to
        drop all Datafiles that are from the same source, as indices will only
        drop the specified Datafile rather than all which match a given label).
    match_strategy : str
        Name of the matching strategy that should be used to find equivalent 
        spatio-temporal locations between Datafile coordinate grids when
        generating samples. For available options and further discussion, see
        the `find_neighbors` docstring in crest/utils/find_neighbors.py. 

    """

    def __init__(self,
        dataset     : Union[Dataset, StructuredDataset, list[Dataset]],
        batch_size  : int,
        epoch_size  : int  | None = None,
        block_size  : int  | None = None,
        features    : list | None = None,
        workers     : int    = 2,
        threads     : int    = 1,
        shuffle     : bool   = True,
        repeat      : bool   = False,
        duplicate   : bool | None = None,
        continuous  : bool   = False,
        block_bytes : Number = 1e9,
        max_queue   : int    = 3,
        task_bytes  : float  = 1e8,
        log_file    : str    = 'Batcher.log',
        log_level   : int    = logging.INFO,
        log_delay   : float  = 0,
        fast_path   : bool   = False,
        block_sync  : bool   = False,
        epoch_sync  : bool   = False,
        numblocks   : int | None = None,
        prefetch    : int | bool = 200,
        prefetcher  : ContextManager | Callable = identity,
        prequeuer   : ContextManager | Callable = identity,
        seed        : int | None = None,
        cache_dir   : Union[Path, str, FSMap, S3Path, None] = None,
        overwrite   : bool = False,
        valid_percents : list[dict[str, dict]] = [],
        drop_datafiles : list[list[str]] = [],
        match_strategy : str = 'multi',
        sample_caching : bool = True, 
    ):
        # Block duplication by default should be the same as repeat
        if duplicate is None:
            duplicate = repeat
            
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.epoch_size  = epoch_size
        self.block_size  = block_size
        self.features    = features
        self.workers     = workers# if shuffle else 0
        self.threads     = threads# if shuffle else 1
        self.shuffle     = shuffle
        self.repeat      = repeat
        self.duplicate   = duplicate
        self.continuous  = continuous and repeat
        self.block_bytes = block_bytes
        self.numblocks   = numblocks
        self.max_queue   = max_queue
        self.task_bytes  = task_bytes
        self.log_file    = log_file
        self.log_level   = log_level
        self.log_delay   = log_delay
        self.fast_path   = fast_path
        self.block_sync  = block_sync
        self.epoch_sync  = epoch_sync
        self.prefetch    = prefetch
        self.prefetcher  = prefetcher
        self.prequeuer   = prequeuer
        self.seed        = seed
        self.cache_dir   = cache_dir
        self.overwrite   = overwrite
        self.valid_percents = valid_percents
        self.drop_datafiles = drop_datafiles
        self.match_strategy = match_strategy
        self.sample_caching = sample_caching
        self.random = random = np.random.default_rng(seed)

        # Verify block config lengths are all the same size (or 0)
        assert( len(valid_percents) in [0, len(drop_datafiles)] or 
                len(drop_datafiles) in [0, len(valid_percents)] ), (
            f'If both {len(valid_percents)=} and {len(drop_datafiles)=}' +
            f' are given, they must contain the same number of items.' )

        # Only one synchronization method can be used
        assert(not (block_sync and epoch_sync)), 'Can only use one sync method'

        # Create a cache for samples from deterministic (sample-limited) blocks
        self._samples_cache = _samples_cache = {}
        
        # Store initialization parameter names for pickling
        # Any variable that is not defined by a local-only name (not self.*)
        #   will not be saved, and will not be initialized on worker processes
        self._init_keys = locals().keys() - {'self', 'prefetcher'}

        # Set the numblocks if None was given
        if numblocks is None:
            try:
                self.numblocks = merge(*dataset.numblocks_dict)
            except AttributeError: 
                self.numblocks = 0
                
        # If multiprocessing, fail quickly when dataset can't be pickled
        if self.workers: self._is_picklable()

        # One worker per dataset in the list, each worker gets all blocks
        if isinstance(dataset, list) and isinstance(dataset[0], Dataset):
            self._handle_dataset_list(dataset)

        # Collect all keys indicating which Datafiles aren't required
        gather_zeros = valfilter(lambda v: np.prod(v.values()))
        zero_percent = map(list, map(gather_zeros, self.valid_percents))
        not_required = set(sum(zero_percent,[]) + sum(self.drop_datafiles,[]))

        # Set Datafile requirements for more uniform sample distribution
        if not_required:
            self.info(f'Updating Datafile.is_required: {not_required=}')
            for df in self.dataset:
                df_vp = df.valid_percent
                index = df.dataset_index
                label = df.label
                match = lambda v: v==index if isinstance(v,int) else v in label
                where = list(filter(match, not_required))

                if where or np.prod(df_vp.values()) == 0:
                    df.is_required = False
                    self.info(f'Not required: {index=} {label=} {where=}')
                else:
                    assert(df.is_required), df
    
        # Cache the Dataset blocks if requested
        if cache_dir is not None:
            self.cache(self.cache_dir, self.overwrite)
            
        
    def __repr__(self):
        return f'Batcher({self.dataset})'

    def __getstate__(self):
        """ Ensures unpickle-able objects are not included """
        return {key: getattr(self, key) for key in self._init_keys}

    def __iter__(self):
        """ Batchers are iterable via their (cached) generator function """
        yield from self.generator()

    def __next__(self):
        """ Next batch can be retrieved via the (cached) generator """
        return next(self.generator())

    def __enter__(self) -> 'Batcher':
        """ Create any necessary resources and start generating batches """
        if self.workers > 0: self._processes
        else:                self._generator
        return self

    def __del__(self):
        """ Release resources prior to Batcher object deletion """
        self.close(0, origin='__del__')

    def __exit__(self, *args, **kwargs):
        """ Release resources upon exit of the context manager """
        self.close(0, origin='__exit__')

    # Logging helper fun
    # @limit_calls(timespan=1)
    def   debug(self, *args, **kwargs): self._log('debug',   *args, **kwargs)
    def    info(self, *args, **kwargs): self._log('info',    *args, **kwargs)
    def warning(self, *args, **kwargs): self._log('warning', *args, **kwargs)
    def   error(self, *args, **kwargs): self._log('error',   *args, **kwargs)


    @cached_property
    def benchmark(self):
        """ Return a Stopwatch function for benchmarking """
        return lambda label, logger=self.debug, **kwargs: Stopwatch(**({
            'message' : f'Batcher.{label}',
            'logger'  : logger,
            # 'silent'  : {'time': 0.05}, # Don't log when time < 0.05 seconds
        } | kwargs))


    def start(self) -> 'Batcher':
        """ Starts generating batches.
        Note that this function is equivalent to calling Batcher.__enter__(),
        and using the context manager API (i.e. `with Batcher() as batches:`)
        in general should be preferred (to handle closing automatically).
        """
        return self.__enter__()

        
    def generator(self, show_timing=False):
        """Top-level function to generate batches.

        The only reason this function would be called rather than iterating
        the Batcher object itself, would be if timings should be logged.

        Parameters
        ----------
        show_timing : bool
            Whether timing logs should be printed to show batch iteration speed

        """
        # Display tqdm progress statistics if requested
        bar_kwargs = {
            'unit_scale': True,
            'smoothing': 0,
            'disable': not show_timing,
            'file': sys.stdout,
        }
        with tqdm(unit=' Batches', position=1, **bar_kwargs) as pbar, \
             tqdm(unit=' Samples', position=0, **bar_kwargs) as pbar2:
            start = time.time()
            for i, batch in enumerate(self._prefetched_batches):
                if show_timing and (i == 0):
                    elapsed = time.time() - start
                    pbar.clear();   pbar2.clear()
                    print(f'\nTime to first batch: {elapsed:.1f} seconds\n')
                    pbar.unpause(); pbar2.unpause()
                pbar2.update(self.batch_size); pbar.update(1)   
                yield tuple(batch) if isinstance(batch, list) else batch

    
    @property
    def is_main_process(self) -> bool:
        """ Spawned processes have _ppid set to the main process PID """
        return not hasattr(self, '_ppid')
        

    def close(self, timeout: Number = 10, origin: str = ''):
        """ Close and delete all thread / process resources """
        # if hasattr(self, '_ppid'):
        #     self.debug(f'Starting exit timer: {timeout=}')
        #     timer = threading.Timer(timeout, lambda: os._exit(0))
        #     timer.start()
        # else: self.debug(f'No _ppid in this process')
            
        class suppress:
            """ contextlib.suppress which is available at interpreter exit """
            def __enter__(self):                  yield
            def __exit__(self, *args, **kwargs):  return True

        # If close was called from __del__, we need to wrap all calls such that
        # exceptions are ignored (as there are no guarantees anything still
        # exists outside of this class, e.g. at interpreter exit)
        handler = (suppress if origin == '__del__' else nullcontext)()
        with handler: self.debug(f'Called close from: {origin} ({timeout=})')

        # Signal threads/workers to exit
        with handler:
            if ('_exit_flag' in self.__dict__) and not self._exit:
                with handler: self.debug(f'Setting exit flag from: {origin}')
                    
                # In certain situations, Event.set can deadlock
                #  (see https://stackoverflow.com/a/73341335/22210498)
                with handler: self._exit_flag.set()

        # Close background thread managers
        for key in ['_block_tasks', '_batch_tasks']:
            with handler:
                if key in self.__dict__:
                    self.__dict__.pop(key).close()

        # Exhaust queue(s) if this is the main process
        for queue_key in ['_queue', '_block_queue']:
            with handler:
                if hasattr(self, queue_key) and self.is_main_process:
                    queue = getattr(self, queue_key)
                    # Discard any queue items if we need to close immediately
                    if timeout == 0:
                        queue.cancel_join_thread()
                    else:
                        while not queue.empty():
                            try:    queue.get_nowait()
                            except: break

        # Only join the prefetching thread if we're in the main thread
        with handler:
            if threading.current_thread() is threading.main_thread():
                if hasattr(self, '_prefetch_thread'):
                    with handler: self._prefetch_thread.join(timeout=timeout)
                    with handler: self.__dict__.pop('_prefetch_thread', None)
                    
        # Delete cached attributes
        for key in ['_generator', '_queue', '_prefetched_batches', '_prefetch_queue', '_block_queue', '_block_queue_lock']:
            with handler: self.__dict__.pop(key, None)

        # Clean up background processes
        for job in self.__dict__.pop('_processes', []):
            pid = code = 'Unknown'
            with handler: pid = job.pid
            with handler: code = job.exitcode
            with handler: self.debug(f'Joining pid={pid} (code={code})')
            with handler:

                # Wait a maximum of `timeout` seconds to join
                try:
                    if timeout: job.join(timeout) or job.close()
                    else:       job.terminate()
                except ValueError:
                    try:        job.terminate()
                    except:     pass
                    self.warning(f'Process {pid} needed to be terminated')
        with handler: self.debug(f'Finished closing Batcher via {origin}')


    @property
    def active_workers(self) -> list[mp.process.BaseProcess]:
        """Helper to check if there are any processes currently running.

        Returns
        -------
        list[mp.process.BaseProcess]
            Returns the current list of running processes, if there are
            any, and an empty list otherwise. This function should be used
            instead of Batcher._processes when checking for running processes,
            since calling Batcher._processes will actually start new processes
            if there aren't any currently running.

        """
        return self.__dict__.get('_processes', [])

    
    @property
    def process_name(self) -> str:
        """ Return a label for the current process """
        return f'Process {getattr(self,"_pidx","Main")} (pid={os.getpid()})'

    
    @property
    def pidx(self) -> int:
        """ Return the process index (or 0 if no index is set) """
        return getattr(self, '_pidx', 0)


    def cache(self, 
        cache_dir  : Union[Path, str, FSMap, S3Path] = 'Cache',
        overwrite  : bool = False, 
        verbose    : bool = True,
        fast_check : bool = False,
    ) -> 'Batcher':
        """Cache blocks in the exact format they will be iterated over.

        By caching the Dataset in the exact format required for creating
        batches, we minimize the number of file reads (e.g. from overlapping
        block regions) as well as chunk the zarr data asymmetrically by
        padding non-symmetric chunks (i.e. optimizing Datafile chunking). Note
        that caching a Dataset object has the side-effect of modifying the 
        underlying Datafiles and Dataset - i.e. a Batcher initialized with
        the same Dataset object that was cached by another Batcher will use
        the previously cached data, even if caching was not explicitly run on
        the second Batcher.

        Parameters
        ----------
        cache_dir : Path | str | FSMap | S3Path
            Location for the cached data to be stored. By default, data is 
            cached in `./Cache`. 
        overwrite : bool
            Flag indicating whether already cached data should be overwritten.
            If False, metadata (but not the data values themselves) are checked
            against the cached data to verify it is the same. This does not
            verify that the data itself is the same, and so changing e.g. one
            of the preprocessor function definitions, might result in the wrong 
            data being used. 
        verbose   : bool
            Additional information printed.
        fast_check : bool
            Allow skipping intensive computation if all datafile hashes are
            available in the cache location. Note that this could result in
            unexpected characteristics for the data being loaded (e.g. with
            a different number of blocks than requested). Also note that this
            functionality is currently disabled for Batcher caching. 

        Returns
        -------
        Batcher
            Returns self. 

        """
        assert(threading.current_thread() is threading.main_thread()), \
            'Batcher caching should only be executed in the main thread'
        assert(self.is_main_process), \
            'Batcher caching should only be executed in the main process'
        
        # Skip if cache folders for all datafile hashes exist
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        elif isinstance(cache_dir, FSMap):
            cache_dir = S3Path(cache_dir)

        self.cache_dir = cache_dir
        names = self.dataset.cache_name
        paths = (cache_dir.joinpath(name) for name in names)
        exist = (p.exists() or '.tiledb' in n for p,n in zip(paths, names))

        if overwrite or not (fast_check and all(exist)):
            with Stopwatch(f'Cached {len(self.dataset)} Datafiles at {cache_dir}\n'):
                self.dataset.generate_samples(**{
                    'cache_path'  : self.cache_dir.joinpath(str(self.log_file).replace('.log','').split('/')[-1]),
                    'block_bytes' : self.block_bytes,
                    'numblocks'   : self.numblocks,
                    'logger'      : self._logger,
                    'shuffle'     : self.shuffle,
                    'verbose'     : verbose, 
                    'optimize'    : True,
                    'compute'     : False, 
                    'overwrite'   : overwrite,
                })
        return self
        
        
    @classmethod
    def load_saved(cls, 
        *args, 
        save_path : str | Path, 
        n_samples : int  = -1,
        verbose   : bool = True,
        **kwargs,
    ):
        """Return context manager for iterating batches saved at a location.
            
        Parameters
        ----------
        *args
            Any positional arguments for the Batcher.
        save_path : str | Path
            Location that batches should be saved to. 
        
        Returns
        -------
        contextmanager
            Context manager enabling drop-in integration where Batcher is
            used. Users can just add '.load_saved' to the standard Batcher
            'with' statement, along with the location to save batches to.

        Examples
        --------
        >>> path = 'SavedBatches/batches.pkl'
        >>> with Batcher.load_saved(dataset, save_path=path) as batches:
        ...    for batch in batches:
        ...        print(batch)

        """

        # Create the pickle if it does not yet exist
        save_path = Path(save_path)
        if not save_path.exists():
            if verbose: logger.info(f'Saving batches to "{save_path}"')
            save_path.parent.mkdir(exist_ok=True, parents=True)

            # Create a Batcher object using the given parameters
            with cls(*args, **kwargs) as batcher:
                assert((n_samples > 0) or (not batcher.repeat)), \
                    'Must give n_samples if repeat=True'

                with Stopwatch(silent=not verbose) as timer:
                    # Load all/n_samples batches from the created Batcher
                    batcher = batcher.generator(show_timing=verbose)
                    batches = ( list(batcher) if n_samples <= 0 else 
                               [next(batcher) for _ in range(n_samples)] )

                    # Store them to the requested save location and log timing
                    try:
                        with save_path.open('wb') as f: pkl.dump(batches, f)                 
                    except: save_path.unlink(missing_ok=True)
                    timer.message = f'Stored {len(batches)} batches'

        # Return the batches wrapped in a context manager to allow load_saved
        # to be dropped in where a Batcher object is normally created and used
        @contextmanager
        def SavedBatches(batches):
            yield batches

        with save_path.open('rb') as f:
            return SavedBatches( pkl.load(f) )


    # Internal functions: shouldn't need to be directly accessed by users
    # ===================================================================
    # 
    # High level flow of batch creation:
    # ----------------------------------
    # Batcher.__iter__ or Batcher.__next__ -> 
    #     generator ->
    #         _iterate_process_batches ->
    #             _queue_batches (if workers > 0)
    #             _generator ->
    #                 _generate_batches ->
    #                     _blocker ->
    #                         Blockset.find_matches
    #                         _batcher ->
    #                             Blockset._parse
    #                             _extract_features
    #                             _finalize_batch
    #                             yield batch

    @cached_property
    def _prefetched_batches(self):
        if not self.prefetch:
            for batches in self._iterate_process_batches():
                try:          yield from batches
                except Empty: pass

            # Close here instead of in _iterate_process_batches
            if self.workers and not self._exit:
                self.close(origin='_prefetched_batches')
            return

        queue_size = self.max_queue 
        if not isinstance(self.prefetch, bool):
            queue_size = int(self.prefetch)

        # This would only happen if we restart the same batcher object
        thread = getattr(self, '_prefetch_thread', None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)
            if thread.is_alive():
                raise Exception('Prefetching thread is hanging')

        # Start workers if they aren't already
        if self.workers > 0: self._processes
        else:                self._generator

        # Enter the prefetcher context manager
        with ensure_with(partial(self.prefetcher, self)) as prefetcher:

            # Start the thread that prefetches batches 
            if getattr(self, '_prefetch_queue', None) is None:
                self._prefetch_queue = Queue(queue_size)
            self._prefetch_thread = threading.Thread(**{
                'target' : self._queue_prefetched, 
                'daemon' : True,
                'name'   : 'prefetcher',
                'kwargs' : {'prefetcher': prefetcher},
            })
            self._prefetch_thread.start()

            # Start yielding batches
            try:
                yield from self._iterate_prefetched()                    
            except KeyboardInterrupt:
                self.info('KeyboardInterrupt')
                self.close(timeout=0, origin='_prefetched_batches') 
                raise
            finally:
                self.debug('Exiting _prefetched_batches')
        

    def _iterate_prefetched(self):
        """ Yield batches from the prefetch queue """
        # Track total number of yielded batches and occasionally log info
        batch_count = start_count = 0
        start_timer = lambda: self.benchmark('_prefetched_batches', self.info).__enter__()
        timer = start_timer()
        running = True
        
        # Keep iterating until the exit signal is set
        while not self._exit:
            try:
                batch = self._prefetch_queue.get_nowait()
                yield batch
                batch_count += 1

                # Keep looping while more batches are being fetched
                running = True
            except Empty: 
                # If no workers alive, loop again for any remaining batches
                if self.workers: 
                    if not (self._processes_alive(verbose=False) or running):
                        break
                else:
                    if not (self._prefetch_thread.is_alive() or running):
                        break
                running = False
                time.sleep(WAIT_TIME)
            
            #if (batch_count % 100) == 0:
            if (time.time() - timer.start['time']) > 60:
                with Stopwatch('timer reset', self.info):
                    indent = lambda s: s.replace('\n\t', '\n\t\t')
                    status = list(map(indent, self._get_status()))
                    nbatch = batch_count - start_count
                    if nbatch == 0:
                        status += [self._get_threads_traceback(return_log=True)]
                    timer.message += '\n\t'.join(['',
                        f'Batches since last status: {nbatch}',
                        f'Total batches: {batch_count}',
                    ] + status)
                    # print(timer.message)
                    timer.__exit__()
                    timer = start_timer()
                    start_count = batch_count

        # Only join the thread here to allow restarting the batcher
        if hasattr(self, '_prefetch_thread'):
            self._prefetch_thread.join(timeout=3)
        self.__dict__.pop('_prefetched_batches', None)

        # Close here instead of in _iterate_process_batches
        if self.workers and not self._exit:
            self.close(origin='_prefetched_batches')

    
    def _get_qsize(self, queue):
        try:    return queue.qsize()
        except: return 0


    def _get_status(self) -> list[str]:
        """ Status report on internal resources """
        status = []
        if hasattr(self, '_queue'):
            status += [f'Worker queue size: ~{self._get_qsize(self._queue)}']
        if hasattr(self, '_prefetch_queue'):
            status += [f'Prefetch queue size: ~{self._get_qsize(self._prefetch_queue)}']
        if hasattr(self, '_prefetch_thread'):
            status += [f'Prefetch thread alive: {self._prefetch_thread.is_alive()}']

        if self.is_main_process and ('_processes' in self.__dict__):
            alive = self._processes_alive(verbose=False)
            status += [f'Workers alive: {len(alive)}/{self.workers}']
            # if not len(alive):
            #     self._exit_flag.set()
        return status
        
    
    def _parse_batch(self, batch):
        """ Allows extending Batcher to handle batch post-processing """
        yield batch


    def _safe_queue_batches(self, queue, batches):
        """ Add batches to a queue without the risk of deadlocking """
        class CheckExit(threading.Timer):
            def __init__(self, exit_flag, *args, check_interval=1, **kwargs):
                super().__init__(*args, **kwargs)
                self.check_interval = check_interval
                self.interval_count = 0
                self.daemon = True
                self.exit_flag = exit_flag
                self.exit = False

            def __enter__(self):
                self.start() 
                return self

            def __exit__(self, *args, **kwargs):
                self.exit = True

            def reset(self):
                self.interval_count = 0
                
            def run(self):
                while not self.finished.wait(self.check_interval):
                    if self.exit or self.exit_flag():
                        break
                    self.interval_count += 1
                    if (self.interval_count % self.interval) == 0:
                        self.function(*self.args, **self.kwargs)
                        
        try:
            with CheckExit(lambda: self._exit, 180, self._get_threads_traceback) as batch_monitor:
                for batch in batches:
                    batch_monitor.reset()
                    count = 0
                    seconds = 5 * 60
        
                    # Ensure exit flag is monitored while waiting on a full queue
                    while not self._exit:
                        try:
                            queue.put_nowait(batch)
                            break
                        except Full:
                            count += 1
                            while (not self._exit) and queue.full():
                                time.sleep(WAIT_TIME)
        
                        # Notify when the queue has been full for a long time
                        if (count % int(seconds*10)) == 0:
                            self.info(f'_safe_queue_batches: {queue=} full ' +
                                      f'for at least {seconds} seconds')
                    if self._exit: break
        except Empty:
            raise
        except Exception as e:
            self.error(f'Exception when queuing batches: {e}\n' + 
                      f'\n{queue=}\n{traceback.format_exc()}')
            self.close(origin='_safe_queue_batches')
            raise
            
    
    def _queue_prefetched(self, prefetcher):
        """ Wraps the generator for background prefetching thread """
        discarded = 0
        for batches in self._iterate_process_batches():
            if not self._exit:
                try:
                    batches = map(prefetcher, batches)
                    self._safe_queue_batches(self._prefetch_queue, batches)
                except Empty: pass
            else: 
                discarded += 1
        if discarded:
            self.info(f'Discarded {discarded} batches due to exit flag')
            
        
    def _processes_alive(self, jobs=None, verbose: bool=True) -> list:
        """ Return a list of worker processes which are alive """
        alive = []
        if '_processes' in self.__dict__:
            for job in (jobs or self._processes):
                if not job.is_alive():
                    if verbose:
                        message = f'Worker {job.pid} exitcode: {job.exitcode}'
                        message+= f' (exit_flag={self._exit})'
                        if job.exitcode and not self._exit:
                            self.error(message)
                        else: self.info(message)
                else: alive.append(job)
        return alive
        
        
    def _iterate_process_batches(self):
        """ Unified interface for single/multiprocessing, yielding batches. """
        try:
            # If using multiprocessing, yield batches from the queue
            if self.workers > 0:

                # Poll the queue for new batches until all workers exit
                jobs = list(self._processes)
                n_empty = 1
                seconds = 120

                while not self._exit:
                    try:          
                        yield self._parse_batch( self._queue.get_nowait() )
                        n_empty = 1
                    except Empty: 
                        # Loop one more time after all workers are finished
                        if not self._processes_alive(jobs) and n_empty > 1:
                            break
                        n_empty += 1
                        time.sleep(0.1)
                    if (n_empty % (seconds * 10)) == 0:
                        self.info(f'_iter_process_batches: No batches from ' +
                                  f'worker queue in at least {seconds} secs')
                        
                    # while not self._exit and self._queue.poll(timeout=0.2):
                    #     yield self._queue.recv()
                    jobs = self._processes_alive(jobs)

                # Exit code 3221225477 is STATUS_ACCESS_VIOLATION, which is
                # commonly caused by an issue with data cached on disk. It can
                # also occur when the system runs out of memory, however.
                self._processes_alive(jobs)

            # Otherwise, just yield from the threaded generator
            else:
                # However, using 'yield from' will result in _generator
                # being prematurely closed, as python will recursively
                # close generators when they are garbage collected; i.e.:
                # - Batcher.generator() reference is garbage collected
                # - so the 'yield from _generator()' is then closed
                # - and so _generator itself is then closed
                # Further discussion: https://stackoverflow.com/a/74923483
                for batch in self._generator:
                    try:          yield self._parse_batch(batch)
                    except Empty: pass
                    
        except KeyboardInterrupt: 
            self.info('KeyboardInterrupt')
            self.close(timeout=0, origin='KeyboardInterrupt') 
            raise
        except Exception as e: 
            self.error(f'Exception: {e}\n{traceback.format_exc()}')

        # Clean up resources once all batches have been yielded
        # Do not wrap in 'finally', as this will also prematurely
        # close the _generator object when Batcher.generator is
        # garbage collected.
        if not (self.prefetch or self.workers):
            self.close(origin='Batcher._iterate_process_batches')


    @cached_property
    def _generator(self):
        """Top-level generator object that performs setup, then yields batches.

        Notes
        -----
        This property, when called, starts the actual generation of batches.
        If it has already been called (and the Batcher has not been closed),
        this property returns the already running batch generator.

        """
        try:
            # self.info(f'{self.process_name} Batcher generator starting')
            process = psutil.Process()
            start_t = time.time()
            dataset = self._datasets[self.pidx] if len(self._datasets) > 1 else self.dataset
            psutil.cpu_percent()

            # Only the first Dataset in the list should be continuous
            if (len(self._datasets) > 1) and self.pidx: 
                self.continuous = False

            # Ensure the lock and flags are created before starting threads
            self._remainder_lock
            self._barrier
            if self.pidx == 0:
                self._first_done.clear()
                self._exit_flag.clear()

            # Samples is a list of dask.Delayed objects or a crest Dataset
            if hasattr(dataset, 'generate_samples'):
                self.debug('Generating dataset blocks...')
                with self.benchmark('_generator', self.info) as timer:
                    blocks = dataset.generate_samples(**{
                        'block_bytes' : self.block_bytes,
                        'numblocks'   : self.numblocks,
                        'compute'     : False, 
                        'verbose'     : True, 
                        'logger'      : self._logger,
                        'shuffle'     : self.shuffle,
                    })
                    timer.message += f' Generated {len(blocks)} dataset blocks'
            else: blocks = dataset

            # Verify there are enough sample blocks if we're not duplicating
            if (len(blocks) < self.workers) and (not self.duplicate):
                self.warning('Not enough sample blocks for workers! ' +
                             'Set Batcher.duplicate=True.')
                if self.pidx >= len(blocks): return

            # Calculate number of samples to be pulled from each block
            if (self.epoch_size or self.block_size):
                if self.epoch_size is not None:
                    spb = (self.epoch_size * self.batch_size) / len(blocks)
                    self.block_size = max(1, int(spb))

                elif self.block_size is not None:
                    bpe = (self.block_size * len(blocks)) / self.batch_size
                    self.epoch_size = max(1, round(bpe))

                if self.pidx == 0:
                    self.info(f'\n\tTo guarantee the full spatiotemporal data '
                            + f'extent has been sampled (an epoch completed), '
                            + f'pull at least {self.epoch_size} batches\n')
                    self.debug(f'samples/block: {self.block_size}   '
                             + f'batches/epoch: {self.epoch_size}')

            # Warn if multiple blocks need to be computed for each batch
            if (self.block_size or self.batch_size) < self.batch_size:
                self.warning(f'Samples/block ({self.block_size}) < '
                    +f'batch size ({self.batch_size}), which means multiple'
                    +' blocks need to be computed for each batch (likely '
                    +'resulting in slow batch generation)')

            # Create threaded task executors for generating blocks and batches
            kwargs = {
                'threads'  : min(len(blocks), self.threads),
                'capacity' : min(len(blocks), self.threads) * 2,
                'exitflag' : (lambda: self._exit),
                'exitset'  : self._exit_flag.set,
                'logger'   : self._logger,
            }
            self._batch_tasks = ThreadedFunction(self._batcher, **kwargs)
            self._block_tasks = ThreadedFunction(self._blocker, **(kwargs|{'capacity':1}))

            divider = ''.join(['-'] * 57)
            message = '\n\t'.join(['', divider, 'Completed epoch'])
            message = f'_generator {message}'

            # If requested, yield batches indefinitely
            while not self._exit:
                    
                with self.benchmark(message, self.info, silent=True) as epoch_log:
                    yield from self._generate_batches(blocks)

                    # Only log a completed epoch if one has actually finished
                    epoch_log.silent = self._exit

                # First worker should check the shared block sample state
                # and warn the user of configs that have limited samples
                if not self._exit:
                    if self.pidx == 0:
                        self._validate_configs()

                # Break the infinite loop if we're not repeating
                # We don't want to set the exit flag here, as that would stop
                #  all of our background process generators, not just this one
                if not self._exit and not self.repeat: break
                
                if self.epoch_sync: 
                    self._synchronize_workers()

                if hasattr(self, '_block_queue_lock'):
                    
                    if (not self.epoch_sync) or (self.pidx == 0):
                        time.sleep(0.25)
                        with self._block_queue_lock:                            
                            try:
                                idx = self._block_queue.get_nowait()
                                self._block_queue.put(idx)
                            except Empty:
                                block_ixs = np.arange(len(blocks)).astype(int)
                                if self.shuffle:
                                    self.random.shuffle(block_ixs)
                                self.info(f'Starting block re-queue of {len(block_ixs)} blocks...')
                                for block_ix in block_ixs:
                                    self._block_queue.put(block_ix)
                                self.info('Finished block re-queue')
                    else:
                        while True:
                            time.sleep(0.5)
                            with self._block_queue_lock:                                
                                try:
                                    idx = self._block_queue.get_nowait()
                                    self._block_queue.put(idx)
                                    break
                                except Empty: pass
                                self.info('Waiting on blocks to be queued...')
            else:
                self.debug('Exiting _generator while loop due to exit flag')

        except KeyboardInterrupt:
            self.info('KeyboardInterrupt')
            self.close(timeout=0, origin='KeyboardInterrupt')
            raise

        except Exception as e:
            message = f'\nException: {e}\n{traceback.format_exc()}\n'
            # message+= f'Forcing halt in 3 seconds...'
            self.error(message)
            self.close(origin='_generator exception')
            raise
            # threading.Timer(3, lambda: os._exit(0)).start()

        finally:
            elapsed = time.time() - start_t
            peakmem = getattr(process.memory_info(), 'peak_wset', -1)
            divider = ''.join(['_'] * 46)
            message = f'{self.process_name} exiting. Resource usage:'
            message += f'\n\t CPU use: {psutil.cpu_percent()}%'
            message += f'\n\tPeak Mem: ' + Stopwatch.readable(peakmem, 'byte')
            message += f'\n\t Elapsed: ' + Stopwatch.readable(elapsed, 'time')
            self.info(f'\n{divider}\n{message}\n{divider}')


    def _generate_batches(self, blocks: list):
        """ Core generator loop which yields batches """
        # Ensure we're only operating on a copy of the list of blocks
        self.n_blocks = len(blocks)
        subsets = list(enumerate(blocks))
        start_t = time.time()

        # No repeating blocks means we use dynamic block allocations in order
        # to provide load balancing; i.e. blocks are allocated to workers via 
        # a shared queue rather than a static subset 
        # static_alloc = self.repeat or self.duplicate or (self.workers <= 1) or self.block_sync
        static_alloc = self.duplicate or (self.workers <= 1) or self.block_sync
        self.info(f'{static_alloc=}: {self.repeat=} {self.duplicate=} {self.workers=}')

        def next_block():
            # Since the block queue indices are based on numblocks rather than
            # the true block count, there may be indices greater than the total
            # number available - so we simply poll until a valid one is found
            with self._block_queue_lock:
                while (idx := self._block_queue.get_nowait()) >= len(subsets): 
                    pass                        
            self.debug(f'{self.pidx=} pulling block {idx}')
            return subsets[idx]
                        
        if static_alloc and self.shuffle:
            self.random.shuffle(subsets)

        # If multiprocessing, use only a subset of the
        # overall blocks unless duplicate was set to True
        if hasattr(self, '_pidx') and not self.duplicate and static_alloc:
            subsets = subsets[self._pidx::self.workers]

        # Initialize a new container for leftover samples
        if not hasattr(self, '_remainder'):
            self._remainder = defaultdict(list)
        n_block = len(subsets)

        # Send off first block ASAP to minimize time to first batch
        if not self._first and (self.pidx == 0):
            if static_alloc:
                first, *subsets = subsets
                self._block_tasks([first])
            else: self._block_tasks([next_block()])
                
        # Wait for first batch job to signal completion
        if not self._first:
            self.info('Waiting for first set to be completed...')

            if self.pidx == 0:
                if self._block_tasks.threads < 1:
                    next(self._block_tasks)
                if self._batch_tasks.threads < 1:
                    while not (len(self._batch_tasks) or self._first or self._exit):
                        time.sleep(WAIT_TIME)
                    for task in self._batch_tasks:
                        yield from task

            message = 'Finished waiting on first set'
            with self.benchmark(f'_generate_batches {message}', self.info):
                while not (self._first or self._exit):
                    time.sleep(WAIT_TIME)

        # Divide them into small subsets to operate over in parallel
        if len(subsets):
            pertask = min(len(subsets), self._subset_blocks.value)
            n_split = max(1, len(subsets) // pertask)
            subsets = np.array_split(subsets, n_split)
            self.info(f'Using {pertask} block/task, {len(subsets)} task total')
        
        # Generator is running if there are any subsets or tasks remaining
        runtime = lambda r: (time.time()-start_t) / max(1, n_block-r)
        running = lambda: (subsets or getattr(self, '_block_tasks', None) or 
                                      getattr(self, '_batch_tasks', None))

        # The number of Block objects currently held in memory should always
        # be <= (number of Datafiles) * (_block_tasks capacity + 1) 
        get_len = lambda d: getattr(d, '__len__', lambda: 1)()
        blk_lim = max(map(get_len, self._datasets)) * self.threads * 3

        # Keep executing until all blocks and batches are processed
        while running() and not self._exit:

            # Fill the queue with data that still needs to be processed
            while subsets and not (self._exit or self._block_tasks.is_full()):
                if static_alloc:
                    self._block_tasks(subsets.pop(0))
                else:
                    # All work is finished once the block queue is empty
                    try:          self._block_tasks(next_block())
                    except Empty: subsets = []

            # Yield any batches that have been generated
            for task in self._batch_tasks:
                yield from getattr(task, 'result', lambda: task)()

            # Clear out any completed block tasks (as no value is returned)
            if self._block_tasks.threads > 0:
                if len(list(self._block_tasks)):
                    remaining = (len(subsets) + len(self._block_tasks)
                        if static_alloc else self._get_qsize(self._block_queue))
                    task_secs = runtime(remaining)
                    tps = Stopwatch.readable(task_secs, 'time')
                    eta = Stopwatch.readable(task_secs * remaining, 'time')
                    self.info(f'{remaining} tasks left @ {tps}/task: ~{eta}\n')

                    # Verify Block objects are being cleared from memory
                    # if len(Block._refs) > blk_lim:
                    #     blk_lim = len(Block._refs) + 1
                    #     self.warning(f'Blocks may not be clearing memory: '+
                    #                  f'{blk_lim-1} block objects held')

            elif len(self._batch_tasks) == 0:
                next(self._block_tasks)

            # Brief sleep to release GIL while waiting for threads
            time.sleep(WAIT_TIME)

        if running(): self.debug('Exiting _generate_batches early (exit flag)')

        # Perform epoch completion tasks and yield any final batches
        if not self._exit:
            yield from self._finish_epoch()

    
    def _finish_epoch(self):
        """ Yield remainder samples as the final batch in an epoch """
        # If we repeat over multiple epochs, save remainders for the next epoch
        if not self.repeat and any(map(len, self._remainder.values())):

            # Get any remaining batches and combine into one batch to yield
            has_any = lambda v: getattr(v, '__len__', lambda: 0)()
            valid_k = [k for k,v in self._remainder.items() if has_any(v)]
            batches = map(list, map(self._remainder.pop, valid_k))
            yield self._finalize_batch(sum(batches, []))


    def _get_block_config(self, blocks, *block_idxs) -> dict:
        class _Config(dict):
            def __init__(self, index, **config):
                super().__init__(**config)
                self.config_index = index
        
        configs = { 'valid_percents' : self.valid_percents or [None],
                    'drop_datafiles' : self.drop_datafiles or [None]}
        configs = [_Config(i, **dict(zip(configs.keys(), vals)))
                    for i,vals in enumerate(zip_longest(*configs.values()))]

        # Give each worker an index on first execution, then increment
        key = '__config_index'
        if not hasattr(self, key):
            setattr(self, key, self.pidx % len(configs))
        setattr(self, key, (getattr(self, key) + 1) % len(configs))
        return blocks, block_idxs, configs[getattr(self, key)]

    
    @cached_property
    def _config_block_count(self):
        """ Shared array storing if a block+config creates limited samples """
        # -1: Configuration has been validated
        #  0: Configuration has not yet been computed
        # >0: Configuration has been computed, but user not yet warned 
        return mp.get_context('spawn').Array(c_int, self.n_total_config)

    
    def _config_index(self, config) -> int:
        """ Index of the configuration within the n_total_config """
        return config.config_index
    
    
    def _config_name(self, index) -> str:
        """ Name of the configuration at the given index """
        return f'Config {index}'

        
    def _samples_cache_key(self, config, block_idxs) -> str:
        """ Return a unique key for the given sample creation configuration """
        return ':'.join(map(str, [*block_idxs, hash(str(config))]))

    
    def _validate_configs(self):
        """ Warn user (once) of any configs producing limited samples """
        if self._config_block_count is not None:
            warning = ''
            for i, count in enumerate(self._config_block_count):
                if count > 1:
                    warning += f'\n{self._config_name(i)} produces only {count-1} samples'
                    self._config_block_count[i] = -1
            
            if warning:
                self.warning(f'{warning}\n')

            # After all configs have been checked and warned, stop monitoring
            if (np.array(self._config_block_count) == -1).all():
                self._config_block_count = None

    
    def _validate_cache(self, config: dict, min_blocks: int = 1):
        """ Warn the user if all blocks for this config produce few samples """
        if not hasattr(self, 'n_blocks'):
            raise Exception('Called _validate_cache before blocks created')

        # Skip if this config has already been checked
        if self._config_block_count is None or not self.sample_caching:
            return
            
        self.debug(f'{list(self._config_block_count)=} {self._config_index(config)=} {config=}')
        if self._config_block_count[self._config_index(config)] != 0:
            return 
            
        index_str = '<block_idx>'
        keys_base = self._samples_cache_key(config, [index_str])
        n_samples = 1 # 0 in _config_block_count means unprocessed

        for idx in range(self.n_blocks):
            key = keys_base.replace(index_str, str(idx))
            val = self._samples_cache.get(key, None)
            
            # Missing key means block is unprocessed or it creates many samples
            if val is None:
                min_blocks -= 1
                if min_blocks <= 0:
                    break
            else:
                # Track total number of samples
                n_samples += len(val if getattr(val, 'make_objs', True) 
                            else val.container[0])
                
        # Fewer than the minimum number of blocks produce many samples
        else:
            idx = self._config_index(config)
            self.info(f'Storing config {n_samples=} for {config=} index={idx}')
            self._config_block_count[idx] = n_samples
            
        
    def _blocker(self, blocks: Collection) -> None:
        """Step 1: Combine the given dataset blocks into a single dask Array.

        Notes
        -----
        This function implements the first of two steps that the Batcher
        performs when generating batches, and is operated in parallel by
        however many threads were requested by Batcher.threads.

        This function is given a collection of dataset blocks, which are
        computed and stacked into a single dask array of (uncomputed) Samples.
        This dask array is then divided into smaller chunks, which are shuffled
        and recombined (based on Batcher.task_bytes) to be further processed by
        the second step (Batcher._batcher). This function does not return
        anything since the intermediate results are passed on to the next
        set of threads as Batcher._batcher tasks.

        Parameters
        ----------
        blocks : Collection
            The collection of dask.delayed dataset blocks that should
            be computed to gather the dask.Array[Sample] objects.

        """

        # Blocks is a list of tuples: [(index, block), ...] 
        try: 
            block_idxs, blocks = zip(*blocks)
        except Exception as e: 
            block_idxs = []
            self.debug(f'Unknown structure: {e} {blocks}')
        
        # Get the configuration to use for this set of blocks, then compute
        blocks, block_idxs, config = self._get_block_config(blocks, *block_idxs)

        if config is None or self._exit:
            self._cleanup_blocks()
            if not self._first: self._first_done.set()
            return
        
        # If this block and config always produce the same samples, skip the
        # entire computation and just use the cached samples
        key = self._samples_cache_key(config, block_idxs)
        if self._samples_cache.get(key, None) is not None:
            self.debug(f'Using sample cache | {key=}: {config=} {block_idxs=}')
            if len(self._samples_cache[key]) == 0:
                self.debug(f'{config=} {block_idxs=} has no samples ({key=})')
                self._cleanup_blocks()
                return
            elif self.sample_caching:
                self._batch_tasks(key, config, block_idxs, continuous=self.continuous)
                return
            
        self.debug(f'Starting _compute_block for {block_idxs=}')
        samples = self._compute_block(blocks, config, block_idxs)

        # If block synchronization was requested, we force workers to wait
        # here until all workers have reached this point
        if self.block_sync: self._synchronize_workers()

        # Exit early if signaled to do so
        task_count = 0
        if not self._exit:

            # Ensure the _first_done flag is set, and Blocks are cleaned up
            if not samples.size:
                samples = self._samples_cache[key] = []                
                self._cleanup_blocks()
                if not self._first: self._first_done.set()
                self._validate_cache(config)
                return

            # Chunk to more reasonable sizes if current chunks are too small
            # if samples.blocks.size < 10:
            #     samples = samples.rechunk((self.batch_size,))
            samples = samples.rechunk((-1,))

            # Note that all of the following logic which handles combining
            # tasks, is now superceded by the handling in Blockset that will 
            # directly generate tasks based on the chosen task_bytes value.
            # Consequently, task_blocks is set to 1 to disable handling here.
            
            # Combine multiple blocks together per task to increase throughput
            numblocks = samples.blocks.size
            block_bytes = samples.nbytes / numblocks
            task_blocks = 1  # self.task_bytes // block_bytes

            # Set the subset blocks across processes
            # This can help to drastically speed up a Batcher that has been 
            # given data which is split into too many blocks. The problem is,
            # if the first block computed is significantly smaller than the
            # others in the data, the amount of memory required per block will
            # be significantly underestimated and subsequent block subsets will
            # be composed of too many blocks. Solution is adaptively modifying
            # the number of blocks per subset (and eventually, bytes per task),
            # but that requires a relatively large rewrite of _generate_batches
            self._subset_blocks.value = int(max(1, task_blocks // numblocks))

            task_blocks = int(max(1, min(numblocks, task_blocks)))
            task_count = ceil(numblocks / task_blocks)
            self.info(f'Adding {task_count} batch tasks: ' +
                      f'{task_blocks} blocks/task, ' +
                      f'~{Stopwatch.readable(block_bytes, "byte")}/block')

            # If requested, shuffle order in which sample blocks are computed
            order = np.arange(samples.blocks.size)
            if self.shuffle: self.random.shuffle(order)
            order = iter(order)

        message = f'Queued {task_count} batch tasks'
        with self.benchmark(f'_blocker {message}', self.info) as timer:

            # Extract batches from blocks in the sample array
            while not self._exit and (idx := list(islice(order, task_blocks))):
                self._batch_tasks(samples.blocks[idx], config, block_idxs, continuous=self.continuous)

                # Change config to None so only the first task updates BlockConfig.sampling_count 
                config = None

                # Wait until the first batch is done, or signaled to exit
                while not ((self._first or self._block_tasks.threads < 1) or self._exit):
                    time.sleep(WAIT_TIME)
                
                # Only use the first sub-block within each block to improve diversity
                if self.block_size is not None:
                    timer.message = f'Queued 1 / {task_count} sub-blocks, discarding remainder'
                    self.debug(f'Queued sub-block slice {idx} for block(s) {block_idxs}')

                    # Manually garbage collect Block object as soon as possible
                    samples = None 
                    self._cleanup_blocks()
                    break

            # Close the background thread manager if exit has been signaled
            if self._exit:
                try:    self._block_tasks.close()
                except: pass
                timer.silent = True


    def _compute_block(self, blocks: Collection, config: dict, block_idxs) -> da.Array:
        """Step 1.1: Compute the samples from the given blocks.

        Notes
        -----
        This is step one in the _batcher function, where the current block of
        Sample objects are converted into lists of the requested features
        (based on the value of Batcher.features).

        Parameters
        ----------
        blocks : Collection
            The collection of dask.delayed dataset blocks that should
            be computed to gather the dask.Array[Sample] objects.
        config : dict
            Dictionary containing the configuration to use when computing the 
            samples. Currently possible keys are {valid_pct, dropped}; see 
            Block.find_matches for further details. 

        Returns
        -------
        da.Array
            dask.Array object containing the samples.

        """

        # Combine multiple blocks into a single array
        message = f'Computed {len(blocks)} blocks via {block_idxs=}'
        with self.benchmark(f'_blocker {message}', self.info) as timer:
            
            # Need to find a better way to do this
            compute = lambda blocks: da.hstack(da.compute(*blocks)) #pyright: ignore
            
            try:
                # Each element in blocks is the Blockset.find_matches function
                samples = compute([find_matches(**{
                    'task_bytes'   : self.task_bytes,
                    'task_samples' : self.block_size,
                    'features'     : self.features if self.fast_path else None,
                    'seed'         : self.seed,
                    'rng'          : self.random,
                    'method'       : self.match_strategy,
                } | config) for find_matches in blocks])

            # But we can handle the case where find_matches was already called
            except TypeError: samples = compute(blocks)

            # Log infomation about the samples that were just computed
            blocks = [None]
            ntotal = Stopwatch.readable(samples.size)
            nbytes = Stopwatch.readable(samples.nbytes, 'byte')
            nblock = samples.blocks.size
            timer.message += f' -> {ntotal} samples ({nblock} blocks, {nbytes})'
        return samples


    def _batcher(self, samples: da.Array | str, config: dict, block_idxs) -> list:
        """Step 2: Separate the given sample array into batches.

        Notes
        -----
        This function implements the second of two steps that the Batcher
        performs when generating batches, and is operated in parallel by
        however many threads were requested by Batcher.threads.

        This function is given a dask array of (uncomputed) Samples, created
        by the first step (Batcher._blocker). This array is:
        1. computed into the actual numpy array of Sample objects
        2. extracted into lists of features (if requested by Batcher.features)
        3. shuffled and then split into batches of numpy object arrays
        4. transformed to the final nested list of feature dicts (if requested)

        Parameters
        ----------
        samples : da.Array
            dask.Array object containing the Samples.

        Returns
        -------
        list
            List of batches, where each batch is either an array of Sample
            objects (when no features are requested); or a nested list of
            feature dictionaries which follows the same nesting structure
            of Batcher.features.

        """

        # Use cached samples if a cache key was given
        if isinstance(samples, str):
            self.info(f'Using cache key {samples}: {config=} {block_idxs=}')
            if not self.sample_caching:
                raise Exception(f'Got cache key with {self.sample_caching=}')
            
            if samples not in self._samples_cache:
                message = f'No samples given, but {samples=} not cached: '
                message+= f'available={list(self._samples_cache)} '
                message+= f'{config=} {block_idxs=}'
                raise Exception(message) 
            samples = self._samples_cache[samples]

            if samples is None:
                raise Exception(f'_batcher: Cache empty for key={samples}')
                
        # Compute the current samples and extract features if requested
        else:
            message = f'Compute/extract {len(samples):,} samples'
            with self.benchmark(f'_batcher {message}', self.debug):
                samples = samples.compute()

                if getattr(samples, 'make_objs', True):
                    samples = self._extract_features(samples)
                    n_samples = len(samples)
                else:
                    n_samples = len(samples.container[0])
                
                # If the number of samples < the requested number per block,
                # it means these are the only samples that can be extracted
                # from the block when using this configuration. Therefore, we
                # can cache the small number of samples and avoid re-computing
                # this block + config every time.
                if self.sample_caching:
                    key = self._samples_cache_key(config, block_idxs)
                    if n_samples < 0.9 * (self.block_size or -1):
                        if key in self._samples_cache:
                            
                            # Should be an error, but only a warning for now (until resolved)
                            if self._samples_cache[key] is None:
                                message = f'Previously found > {self.block_size*0.9}'
                                message+= f' samples for {config=} {key=}'
                                self.warning(message)
                                
                                message = f'Re-caching samples for {config=} {block_idxs=} '
                                message+= f'via {key=} ({n_samples=} < {self.block_size=})'
                                self.info(message)
                                self._samples_cache[key] = samples
                                self._validate_cache(config)
                            else:
                                assert(len(self._samples_cache[key]) == len(samples)),\
                                    f'{len(self._samples_cache[key])} != {len(samples)}'
                        else:
                            message = f'Caching samples for {config=} {block_idxs=} '
                            message+= f'via {key=} ({n_samples=} < {self.block_size=})'
                            self.info(message)
                            self._samples_cache[key] = samples
                            self._validate_cache(config)
                    else: self._samples_cache[key] = None
        
        if self._exit:
            self.debug(f'Exiting _batcher early due to exit flag')
            return

        def create_batches(samples, remainder, put_in_queue=False):
            if 0 < len(remainder) < self.batch_size:
                samples = np.append(remainder, samples, 0)

            splits  = np.arange(self.batch_size, len(samples), self.batch_size)
            samples = np.array_split(samples, splits)

            if put_in_queue:
                for batch in map(self._finalize_batch, samples):
                    self._queue.put(batch)
                return [], []

            if len(samples[-1]) < self.batch_size:
                *samples, remainder = samples
            else: remainder = []
            return samples, remainder


        with self.benchmark('_batcher', self.info) as gen_timer:
                
            # Get a remainder key that is used across all block indices
            conf_hash = self._samples_cache_key(config, [-1])
            remainder = self._remainder[conf_hash]
            
            # Fast path which does not create Sample objects
            if not getattr(samples, 'make_objs', True):
                self._dtypes = samples.ele_dtype
                samples = samples.container

                if self.shuffle:
                    index = np.arange(len(samples[0]))
                    self.random.shuffle(index)
                    samples = [s[index] for s in samples]

                # Use a lock to ensure _remainder is handled by only one thread
                with self._remainder_lock:
                    samples, remainder = zip( *map(create_batches,
                        samples, remainder or ([[]] * len(samples))) )

                    if len(remainder[0]):
                        self._remainder[conf_hash] = remainder
                samples = list(map(self._finalize_batch, zip(*samples)))

            else:
                if self.shuffle: self.random.shuffle(samples)

                # Faster, but discards any (samples % batch_size) samples
                if self.continuous:
                    samples, _ = create_batches(samples, remainder, True)
                else:

                    # Use a lock to ensure _remainder is handled by only one thread
                    with self._remainder_lock:
                        samples, self._remainder[conf_hash] = create_batches(samples, remainder)

                        # # Calculate attributes to create a view of the samples
                        # n_group = len(samples) // self.batch_size
                        # i_size  = samples.itemsize
                        # shape   = (n_group, self.batch_size) + samples.shape[1:]
                        # strides = (self.batch_size*i_size,i_size) + samples.strides[1:]

                        # # Save any remainder samples
                        # self._remainder = samples[n_group * self.batch_size:]

                    # Create a strided view and create the batch feature dicts
                    # samples = as_strided(samples, shape=shape, strides=strides)
                    # list(map(self._queue.put, map(self._finalize_batch, samples)))
                    # list(map(self._queue.send, map(self._finalize_batch, samples)))
                    samples = list(map(self._finalize_batch, samples))

            # Signal that the first batch set is now completed
            if not self._first: self._first_done.set()
            self._cleanup_blocks()
            gen_timer.message += f' Generated {len(samples)} batches'
            return samples


    def _extract_features(self, samples: Union[SampleSet, np.ndarray]) -> np.ndarray:
        """Step 2.1: Extract the requested features from the Sample objects.

        Notes
        -----
        This is step one in the _batcher function, where the current block of
        Sample objects are converted into lists of the requested features
        (based on the value of Batcher.features).

        Parameters
        ----------
        samples : SampleSet | np.ndarray
            The current block of samples the thread is working on. This can
            be either a raw SampleSet object, or the equivalent numpy array,
            depending on how dask internally handled generating the block.

        Returns
        -------
        np.ndarray
            Returns a numpy array with the same length as the given `samples`
            object. This is either the same exact object (if no value was
            given for Batcher.features), or a nested numpy object array which
            follows the same nesting structure as Batcher.features.

        """
        # If no features were requested, return the samples unmodified
        if (self.features is None) or (len(samples) == 0): return samples[:]

        # If an empty list was passed in, all available features are requested
        if len(self.features) == 0: self.features = samples[0].features

        def _extract(sample, features):
            """ Follow the nesting structure given by the features """
            if isinstance(features[0], str) or not hasattr(features[0], '__len__'):
                return type(features)(sample.to_list(features))
            return type(features)(map(partial(_extract, sample), features))
        return np.frompyfunc(partial(_extract, features=self.features), nin=1, nout=1)(samples)


    def _finalize_batch(self, batch: np.ndarray) -> Union[np.ndarray, dict, list]:
        """Step 2.2: Transform the batch array into the final feature dict.

        Notes
        -----
        This is step two in the _batcher function, where the batch of Sample
        objects are converted into the requested format:
            - numpy array of Sample objects (if Batcher.features = None)
            - feature dictionaries (if Batcher.features = list of strings)
            - nested list of dicts (if Batcher.features = nested list of str)

        Parameters
        ----------
        batch : np.ndarray
            The batch of Sample objects to extract from.

        Returns
        -------
        np.ndarray | dict | list
            The return type depends on what value the `features` parameter
            was given when the Batcher object was created:
            - if nothing was given, Batcher.features=None, and the batch array
              is returned unmodified (and so will be a numpy array of Samples)
            - if a list of strings was given for `features`, the return value
              will be a dictionary in which the keys are the requested features
              and the values are the numpy arrays shaped [batch size, ...] for
              each requested feature
            - if a nested list of strings was given (e.g. [['f1'], ['f2']]),
              then the returned object will be a nested list of dictionaries
              with the same structure as the requested features (e.g.
              [{'f1': <np.ndarray>}, {'f2': <np.ndarray>}])

        """

        if getattr(self, '_dtypes', None) is not None:
            batch: dict = {feature: data[:, i].astype(T) 
                for data, dtype in zip(batch, self._dtypes, strict=True)
                for i, (feature, T) in enumerate(dtype.items())}
            def _nest(features):
                """ Follow the nesting structure given by the features """
                if isinstance(features[0], str) or not hasattr(features[0], '__len__'):
                    return {f: batch[f] for f in features}
                return type(features)(map(_nest, features))
            return _nest(self.features) if self.features else batch

        def _parse(batch, features):
            """ Follow the nesting structure given by the features """
            if isinstance(features[0], str) or not hasattr(features[0], '__len__'):
                return dict(zip(features, map(np.array, zip(*batch))))
            return type(features)(map(_parse, zip(*batch), features))
        return batch if self.features is None else _parse(batch, self.features)
        

    def _queue_batches(self,
       process_ix : int,
       shared_attrs : dict,
       # parent_pid : int,
       # queue      : mp.queues.Queue,
       # exit_flag  : mp.synchronize.Event,
       # first_done : mp.synchronize.Event,
       # sub_blocks : mp.sharedctypes.Synchronized,
       # barrier    : mp.synchronize.Barrier,
    ) -> None:
        """Helper to put batches into a multiprocessing queue.

        Notes
        -----
        This function is the entry point for worker processes.

        Parameters
        ----------
        process_ix : int
            Integer specifying which worker process this is (ranging from
            0 to `workers-1`). This is used to select which subset of dataset
            blocks to operate on when Batcher.duplicate is set to False.
        parent_pid : int 
            PID of the process which spawned this child process. Used in order
            to handle the edge case of a parent process being killed outside
            of the Python handling framework, which would otherwise cause any
            child processes to become orphans. 
        queue      : multiprocessing.queues.Queue
            The multiprocessing Queue object that workers will put batches
            into. The main process will monitor this queue to receive batches
            and yield them to the rest of the program.
        exit_flag  : multiprocessing.synchronize.Event
            Event used to signal the process to exit.
        first_done : multiprocessing.synchronize.Event
            Event used to signal that the first batch has been generated.
        sub_blocks : mp.sharedctypes.Synchronized
            Shared value used to indicate how many dataset blocks should be 
            contained in each _blocker task.

        """
        # UploadMonitor is required to import in order to register it
        # Since this method is the entry point for new processes, it's the
        #   easiest spot to import and register the class. It doesn't really
        #   make sense to import it here from an organizational standpoint
        #   though, so it does need to be moved somewhere else eventually.
        # Note that UploadMonitor should no longer be necessary to import
        # and register anymore, and it is only kept here (for now) in order
        # to allow Batcher to work with older datasets.
        try:
            from terrahydro.data.utils.UploadMonitor import UploadMonitor
        except:
            pass

        # Make sure we catch and log any exceptions, as they will disappear
        #  silently otherwise (since we're in a background process here)
        try:
            # Ensure we are properly catching any segmentation faults
            # faulthandler.enable()

            # This function only runs inside a worker process, which means
            # KeyboardInterrupts should be ignored (as it will be caught and 
            # handled by the main process, which can signal through _exit 
            # if this child should actually exit). 
            # Without this set, the Batcher workers will _always_ receive a 
            # KeyboardInterrupt signal when ctrl-c is pressed, even if the 
            # main process that's using the Batcher is gracefully catching it,
            # and wants to continue execution and generate more batches.
            # https://stackoverflow.com/a/5050521
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            # Store the process metadata and shared objects
            self.__dict__.update(shared_attrs | {'_pidx': process_ix})
            queue = shared_attrs['_queue']

            # When duplicate is set, generate new randomness per process
            if self.duplicate: self.random = np.random.default_rng(process_ix)
            self.debug(f'{self.process_name} initialized')

            # import memray
            # with memray.Tracker(f'output.bin.{os.getpid()}'):

            # Start adding batches to the queue
            with ensure_with(partial(self.prequeuer, self)) as prequeuer:
                prequeuer = self._map_to_batch(prequeuer)
                self._safe_queue_batches(queue, map(prequeuer, self._generator))

        except Exception as e:
            self.error(f'{self.process_name} exception: {e}\n' +
                       f'{traceback.format_exc()}')
            raise

        # Ensure queued items do not block this process from joining
        finally:
            if self._exit: self.close(0, origin='_queue_batches')
            self.debug(f'{self.process_name} waiting for queue to clear')
            while len(queue._buffer):
                if not psutil.pid_exists(self._ppid):
                    queue.cancel_join_thread()
                time.sleep(WAIT_TIME)
            self.info(f'{self.process_name} exiting')


    def _map_to_batch(self, function):
        """ Allows wrapping a function that is applied to pre-queue batches """
        return function

        
    def _is_picklable(self) -> None:
        """ Checks that the dataset is picklable before spawning processes """
        try:
            pkl.dumps(self.dataset)
        except pkl.PicklingError:
            message = f'Dataset {self.dataset} is required to be picklable '
            message += 'when using multiprocessing (i.e. Batcher.workers > 0)'
            self.error(message)
            raise Exception(message)


    def _handle_dataset_list(self, datasets: list[Dataset]) -> None:
        """ Configure to handle a list of Dataset objects, one per worker """
        if len(datasets) > 1:
            if self.workers and (self.workers != len(datasets)):
                raise Exception(f'Batcher.workers={self.workers}, but a list '+
                    f'with {len(datasets)} Datasets was given. The number of '+
                    f'workers must be either 0 (for non-multiprocessing), or '+
                    f'{len(datasets)} (to allow one Dataset per worker).')
            self.duplicate = True # Workers get all blocks in their Dataset
        else: self.dataset = datasets[0]


    def _cleanup_blocks(self) -> None:
        """ Manually garbage collect Block object as soon as possible """
        # message = f'GC: {len(Block._refs)}'
        # with self.benchmark(f'_cleanup_blocks {message}', self.debug) as timer:
        #     gc.collect()
        #     timer.message += f' -> {len(Block._refs)} Block references'


    @property
    def _shared_attrs(self) -> dict:
        """ Returns a dictionary of attributes shared across processes """
        return {
            '_ppid'          : os.getpid(),
            '_exit_flag'     : self._exit_flag,
            '_first_done'    : self._first_done,
            '_subset_blocks' : self._subset_blocks,
            '_barrier'       : self._barrier,
            '_queue'         : self._queue,
            '_block_queue'   : self._block_queue,
            '_block_queue_lock' : self._block_queue_lock,
            '_config_block_count' : self._config_block_count,
        }

    
    def get_queue(self, context, max_queue=None):
        """ Allows extending the type of queue used for batching """
        return context.Queue(max_queue or self.max_queue)

        
    @cached_property
    def _processes(self) -> list[mp.process.BaseProcess]:
        """Create worker background processes.

        Notes
        -----
        - Processes are created in a spawn context, regardless of if the
          operating system supports forking.
        - Another way to implement this method is via a multiprocessing.Pool
          or concurrent.futures.ProcessPoolExecutor object. Both of these
          require using a multiprocessing.Manager to create the queue however,
          and the overall performance is a fair bit slower than using manually
          created multiprocessing.context.SpawnProcess objects (as done below).

        Returns
        -------
        list[multiprocessing.process.BaseProcess]
            A list of started background processes which are operating on
            the Batcher._queue_batches method. These background processes
            should not be interacted with directly, as their generated
            batches can be accessed via the Batcher._queue object.

        """
        self.info(f'Starting {self.workers} background processes')
        assert(threading.current_thread() is threading.main_thread()), \
            'Batcher worker processes can only be started in the main thread' 
        self._exit_flag.clear()

        # Create the spawning context and the queue used to transfer batches
        mp_context = mp.get_context('spawn')
        self._queue = self.get_queue(mp_context)
        # self._queue, conn = mp_context.Pipe(False)

        numblocks = self.numblocks
        if isinstance(numblocks, dict):
            options = self.dataset.dims[0]
            unknown = [d for d in numblocks if d not in options]
            assert(len(unknown) == 0), f'Dims {unknown=}; {options=}'
            numblocks = [numblocks.get(d, 1) for d in options]
        
        n_blocks = int(np.prod(np.atleast_1d(numblocks)))
        self._block_queue = self.get_queue(mp_context, n_blocks+1)
        self._block_queue_lock = mp.get_context('spawn').Lock()
        
        # If we're not repeating blocks, add indices to the block queue
        # to enable dynamic block allocation for workers
        if True:#not self.repeat:
            block_ixs = np.arange(n_blocks).astype(int)
            if self.shuffle:
                self.random.shuffle(block_ixs)
            for block_ix in block_ixs:
                self._block_queue.put(block_ix)
            
        # Create and start the background processes
        kwargs = {'target': self._queue_batches, 'daemon': True}
        create = lambda i: mp_context.Process(args=(i, self._shared_attrs), **kwargs)
        jobs = [p.start() or p for p in map(create, range(self.workers))]
        assert (len(jobs) == self.workers), jobs

        # Guarantee child processes exit instead of being orphaned
        def handle_signal(original_handler, _jobs=list(jobs)):
            """ Ensures exit signals are handled correctly and stop workers """
            count = 0
            def handle(sig_id, frame, function=original_handler, jobs=_jobs):
                nonlocal count
                count += 1
                signals = {getattr(x, 'value', x): x for x in signal.valid_signals()}
                sig = signals.get(sig_id)
                if count <= 1:
                    message = f'Received {sig.name} from {frame=} stack:\n'
                    message+= ''.join(traceback.format_list(traceback.extract_stack(frame)))
                    message+= '\nAttempting graceful exit...\n'
                    try:    self.error(message)
                    except: logger.error(message)
                    function(sig_id, frame)
                else:
                    message = f'Received {sig.name} {count} times; '
                    message+= 'Halting immediately\n'
                    try:    self.error(message)
                    except: logger.error(message)
                    while len(jobs):
                        try:
                            message = f'Terminating {jobs[-1]}'
                            try:    self.error(message)
                            except: logger.error(message)
                            try: jobs[-1].terminate()
                            except: pass
                            jobs.pop()
                        except: pass
                    sys.exit(0)
            return handle

        for sig in [signal.SIGINT, signal.SIGTERM]: #signal.valid_signals():
            signal.signal(sig, handle_signal(signal.getsignal(sig)))

        # Wait a second for them to start, then ensure that they are running
        time.sleep(1)
        if any(not job.is_alive() for job in jobs):
            message = 'Batcher processes are stopping immediately. This is '
            message += 'possibly due to issues pickling the given Dataset, if '
            message += 'no other exceptions are logged.'
            message += f'\nExit codes: {[job.exitcode for job in jobs]}'
            self.warning(message)
            if not any(job.is_alive() for job in jobs):
                raise Exception(message)
        return jobs


    @cached_property
    def _exit_flag(self) -> mp.synchronize.Event:
        """Flag to signal Batcher exit.

        Returns
        -------
        multiprocessing.synchronize.Event
            This uses multiprocessing.Event in order to allow signaling across
            background worker processes as well as intra-process threads.

        """
        return mp.get_context('spawn').Event()
        

    @property
    def _exit(self) -> bool:
        """ Shortcut to check if the exit flag has been set """
        orphaned = not (self.is_main_process or psutil.pid_exists(self._ppid))
        return self._exit_flag.is_set() or orphaned


    @cached_property
    def _first_done(self) -> mp.synchronize.Event:
        """Flag to signal the first batch has been generated.

        Notes
        -----
        Forcing all threads/workers to wait for the first batch to be produced
        allows a faster time to first batch, since there is no competition
        for computational resources for the single thread generating it.

        Returns
        -------
        multiprocessing.synchronize.Event
            This uses multiprocessing.Event in order to allow signaling across
            background worker processes as well as intra-process threads.

        """
        return mp.get_context('spawn').Event()


    @property
    def _first(self) -> bool:
        """ Shortcut to check if the first batch has been computed """
        # If using block synchronization, waiting on the first block is skipped
        if self.block_sync:
            return True
        return self._first_done.is_set()


    @cached_property
    def _barrier(self) -> mp.synchronize.Barrier:
        """ Multiprocessing Barrier.

        Synchronization object that prevents workers from moving forward until
        all workers have reached and called the Barrier.
        
        Returns
        -------
        multiprocessing.synchronize.Barrier
            Only used to synchronize workers across processes such that workers
            will wait until all other workers have completed their block before
            moving on to a new block.

        """
        return mp.get_context('spawn').Barrier(max(1, self.workers))


    def _synchronize_workers(self):
        """ Block until all workers have called this function.

        This function creates a new thread to monitor the exit signal, so that 
        if signaled the timer thread will abort the barrier to allow exiting.

        """
        class CheckExit(threading.Timer):
            def run(T):
                while not T.finished.wait(T.interval):
                    if self._exit:
                        self._barrier.abort()
                        break
        # Check the exit signal once per second
        timer = CheckExit(1, lambda: None)
        timer.start()
        if not self._barrier.broken:
            left = self._barrier.parties - (self._barrier.n_waiting + 1)
            name = self.process_name
            if left: self.info(f'{name}: {left} more workers to synchronize..')
            else:    self.info(f'{name} is last to synchronize')
        else: self.info(f'{self.process_name} encountered a broken barrier')      
        try: self._barrier.wait()
        except threading.BrokenBarrierError:
            if not self._exit: raise
        finally: timer.cancel()        
            

    @cached_property
    def _subset_blocks(self) -> mp.sharedctypes.Synchronized:
        """ Synchronized value across processes which defines subset blocks """
        shared_value = mp.get_context('spawn').Value(c_int)
        shared_value.value = 1
        return shared_value


    @cached_property
    def _remainder_lock(self) -> threading.Lock:
        """ Lock ensuring _remainder is modified by one thread at a time """
        return threading.Lock()


    @cached_property
    def _datasets(self) -> list[Dataset] | list[StructuredDataset]:
        """ Circumvents the need to check if dataset is a list everywhere """
        if isinstance(self.dataset, list):
            if isinstance(self.dataset[0], (Dataset, StructuredDataset)):
                return self.dataset
        return [self.dataset]

    
    @property
    def n_total_config(self) -> int:
        """ Number of block configurations """
        return max(len(self.valid_percents), len(self.drop_datafiles), 1)

        
    def _get_threads_traceback(self, limit=4, names=[], return_log=False):#'MainThread', '_batcher', '_blocker']):
        """ Prints the stack trace for active threads """
        log = f'\n{self.process_name} Threads Status'
        bar = '_' * len(log)
        div = '-' * len(log)
        log+= f'\n{bar}\n'
        
        for tid, frame in sys._current_frames().items():
            # Get the thread object corresponding to the frame
            for t in threading.enumerate():
                if t.ident == tid:

                    # If names are given, only log those requested threads
                    if names and not any(n in t.name for n in names):
                        continue

                    # Skip irrelevant tracebacks
                    trace = ''.join(traceback.format_stack(frame, limit=limit))
                    skips = [
                        'work_item = work_queue.get(block=True)',
                        'waiter.acquire()',
                        'wacquire()',
                        '.join(traceback.format_stack(frame, limit=limit))',
                    ]
                    if not any(map(trace.strip().endswith, skips)):
                        log += f'Thread: {t.name} (ID: {tid})\n{trace}{div}\n'
                    break
        if return_log:
            return log
        self.info(log)

    
    @cached_property
    def _logger(self):
        """ Create the logging object which writes logs to a file """
        format_key = [
            '%(asctime)23s',
            '%(process)6s',
            '%(threadName)10.10s',
            '%(lineno)4s:%(filename)-9.9s..',
            '%(levelname)7s',
            '%(message)s',
        ]
        log_format = logging.Formatter(' | '.join(format_key))

        logger = logging.getLogger(Path(self.log_file).name)
        if logger.hasHandlers():
            logger.handlers.clear()

        # abseil hijacks the root logger
        logger.propagate = False
        
        # If a log_file is requested, use a file handler
        if self.log_file is not None:

            # Add a handler for error logs that also prints to sys.stderr
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            handler.setFormatter(log_format)
            logger.addHandler(handler)

            # Define the log header
            replace = {
                'process'   : 'pid',
                'levelname' : 'level',
                'lineno'    : 'lineNum',
            }
            pattern = re.compile(r'\((.*?)\)')
            labels = list(chain(*map(pattern.findall, format_key)))
            values = [replace.get(label, label) for label in labels]
            inserts = ' | '.join(format_key) % dict(zip(labels, values))
            headrow = inserts.replace('.. ', '')
            divider = re.sub(r'[^|]', '=', headrow)
            header = f'{headrow}\n{divider}\n'
            
            # If we're in the main process, delete the prior log_file
            if not mp.current_process().daemon:
                filename = Path(self.log_file).absolute()
                if not filename.parent.exists():
                    filename.parent.mkdir(parents=True)
            
                # Backup previous log file if it exists
                if filename.exists():

                    # Remove any existing backups
                    for backup in filename.parent.glob(f'{filename.name}*.backup'):
                        try:
                            backup.unlink()
                        except Exception as e:
                            logger.warning(f'Exception removing file {backup}: {e}')

                    # Make backup of main Batcher.log, as well as any process logs
                    for plog in filename.parent.glob(f'{filename.name}.*'):
                        shutil.copy(plog, f'{plog.as_posix()}.backup')
                        try:
                            plog.unlink()
                        except Exception as e:
                            logger.warning(f'Exception removing file {plog}: {e}')

                    shutil.copy(filename, Path(f'{filename.as_posix()}.backup'))
                Path(self.log_file).write_text(header)
            
            # Otherwise, add a second handler specifically for this process
            elif self.workers > 1:
                filename = Path(self.log_file).as_posix() + f'.{os.getpid()}'
                handler2 = logging.FileHandler(filename)
                Path(filename).write_text(header)
                handler2.setFormatter(log_format)
                logger.addHandler(handler2)

            # A rotating handler would be nice, but doesn't work in Windows
            #   Note that if a rotating handler does ever get added, we need to
            #   make sure the first log file generated is always kept since it
            #   will contain various configuration info useful for debugging
            handler = logging.FileHandler(self.log_file)

        # Otherwise just log to sys.stderr
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(log_format)

        # Use a time-based MemoryBuffer to allow logs generated by 
        # different worker processes to be grouped together in the log
        # file when they are within `log_delay` seconds of each other.
        #   Note that when this is used, it means the log file might have
        #   logs that are out of order temporally (by <= `log_delay` secs)
        if (self.workers > 1) and (self.log_delay > 0):
            handler = TimedHandler(self.log_delay, capacity=50, target=handler)

        logger.setLevel(self.log_level)
        logger.addHandler(handler)
        return logger


    def _log(self, method, *args, **kwargs):
        """ Helper which allows modifying the stacklevel for correct labels """
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 2
        getattr(self._logger, method)(*args, **kwargs)