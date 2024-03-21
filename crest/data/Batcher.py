from numpy.lib.stride_tricks import as_strided
from collections.abc import Collection
from dask.delayed import Delayed
from contextlib import nullcontext
from functools import partial, cached_property
from itertools import chain, islice
from tqdm.auto import tqdm
from numbers import Number
from pathlib import Path
from ctypes import c_int
from typing import Union
from queue import Empty, Full
from math import ceil

import multiprocessing.sharedctypes
import multiprocessing.synchronize
import multiprocessing.queues
import multiprocessing as mp
import threading, _thread
# import tensorflow as tf
import dask.array as da
import pickle as pkl
import numpy as np
import faulthandler
import traceback
import logging
import psutil
import shutil
import time
import os
import re
import gc

from crest.utils import Stopwatch
from .ThreadedFunction import ThreadedFunction
from .loading import Dataset, StructuredDataset, SampleSet, Block

from crest.utils.setup_logging import logger_setup

logger = logging.getLogger(__name__)

# Amount of time to sleep when waiting on something to complete
#   This should be a small value, but large enough that there won't be
#   a large number of GIL releases which can cause additional slowdown
WAIT_TIME = 0.1


class Batcher:
    """Handles creating batches of data samples.

    Parameters
    ----------
    dataset    : Dataset | StructuredDataset
        Crest Dataset or StructuredDataset object.
    batch_size : int
        Number of samples that each batch should contain.
    features   : list
        List of features that should be extracted from batch Samples. By
        default, no features are extracted, and a batch will be a list of
        Sample objects. When given, a batch will have the same nested layout
        as the feature list, and contain dictionaries with features as keys
        and numpy arrays (shaped [batch_size, ...]) as values. For example:
        features=[['a', 'b'], ['c']] would result in batches that look like
        
        - [{'a':<batch of 'a' values>, 'b': <batch of 'b' values>},
           {'c': <batch of 'c' values>}]

        If any features are missing in the
        dataset, an exception is raised. If an empty list is given, it is
        equivalent to selecting all available features. Note that any nested
        list in features must contain homogeneous types; i.e. all strings, or
        all lists:
        
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
        the original order of the sample array. When shuffle=False, multiple
        processes and threads may not be used, as completion progress across
        processes cannot be consistently ordered.
    repeat     : bool
        Whether the Batcher should repeat iteration over the batches once all
        batches have been yielded. Note that this means the Batcher will yield
        batches indefinitely.
    duplicate  : bool
        Whether batch samples across processes can be duplicated. With this set
        to True, each sample will be encountered `worker` times in an epoch,
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
    blocksize  : Number
        Number of bytes that should be allocated to each block and worked
        on in parallel (default=1e8; 100MB). Note that this is just a proxy
        for the amount memory that will be used when computing a block, as 
        the actual amount used is dependent upon the number of matches that
        are found in the block and thus can vary significantly. 
    numblocks  : int 
        Alternative to giving a blocksize value. If numblocks > 0, the 
        requested number of blocks is the target block total. While the 
        exact number of blocks is not always possible to create, an attempt
        is made to get as close as possible to the requested value.
    max_queue  : int
        Number of batches to generate in advance when using multiprocessing.
        When `workers` <= 0, has no effect.
    task_bytes : float
        Maximum number of bytes that should comprise a _batcher task. This
        value controls the tradeoff between memory usage and batch throughput,
        with a higher number of bytes per task leading to more batches created
        in a single task. Having a value too large can cause long periods
        without batches however, even if the average throughput is marginally
        greater. In general, larger individual samples (those with large
        windows in multiple dimensions, for instance) will require larger
        bytes per task in order to achieve optimal throughput.
    logfile    : bool
        File that logs should be written to.
    loglevel   : int
        Level that log file will display. Should be a level defined by the
        logging module, i.e. logging.INFO, logging.DEBUG, etc.
    seed       : int | None
        Seed for reproducible randomness.

    """

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __init__(self,
        dataset    : Union[Dataset, StructuredDataset, list[Dataset]],
        batch_size : int,
        features   : list | None = None,
        workers    : int    = 2,
        threads    : int    = 1,
        shuffle    : bool   = True,
        repeat     : bool   = False,
        duplicate  : bool   = False,
        continuous : bool   = False,
        blocksize  : Number = 1e9,
        numblocks  : int    = 0,
        max_queue  : int    = 100,
        task_bytes : float  = 5e7,
        logfile    : str    = 'Batcher.log',
        loglevel   : int    = logging.INFO,
        seed       : int | None = None,
    ):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.features = features
        self.workers = workers if shuffle else 0
        self.threads = threads if shuffle else 1
        self.shuffle = shuffle
        self.repeat = repeat
        self.duplicate = duplicate
        self.continuous = continuous and repeat
        self.blocksize = blocksize
        self.numblocks = numblocks
        self.max_queue = max_queue
        self.task_bytes = task_bytes
        self.logfile = logfile
        self.loglevel = loglevel
        self.seed = seed
        self.random = random = np.random.default_rng(seed)

        # Store initialization parameter names for pickling
        self._init_keys = locals().keys() - {'self'}

        # If multiprocessing, fail quickly when dataset can't be pickled
        if self.workers: self._is_picklable()

        # One worker per dataset in the list, each worker gets all blocks
        if isinstance(dataset, list) and isinstance(dataset[0], Dataset): 
            self.workers   = len(dataset)
            self.duplicate = True


    def __getstate__(self):
        """ Ensure unpickle-able objects are not included """
        return {key: getattr(self, key) for key in self._init_keys}

    def __del__(self):
        self.close(0, origin='__del__')

    def __iter__(self):
        yield from self.generator()

    def __next__(self):
        return next(self.generator())

    def __exit__(self, *args, **kwargs):
        self.close(0, origin='__exit__')

    def __enter__(self):
        """ Create any necessary resources and start generating batches """
        if self.workers > 0:
            self._processes
        else:
            self._generator
        return self

    def debug(self, *args, **kwargs):
        self._log('debug', *args, **kwargs)

    def info(self, *args, **kwargs):
        self._log('info', *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._log('warning', *args, **kwargs)

    def error(self, *args, **kwargs):
        self._log('error', *args, **kwargs)

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
        }
        with tqdm(unit=' Batches', **bar_kwargs) as pbar, \
                tqdm(unit=' Samples', **bar_kwargs) as pbar2:
            start = time.time()
            for i, batch in enumerate(self._iter_process_batches()):
                if show_timing and (i == 0):
                    elapsed = time.time() - start
                    pbar.clear();
                    pbar2.clear()
                    self.logger.info(f'Time to first batch: {elapsed:.1f} seconds\n')
                    pbar.unpause();
                    pbar2.unpause()
                pbar.update(1);
                pbar2.update(self.batch_size)
                yield batch

    def close(self, timeout: Number = 10, origin: str = ''):
        """ Close and delete all thread / process resources """

        class suppress:
            """ contextlib.suppress which is available at interpreter exit """

            def __enter__(self):                  yield

            def __exit__(self, *args, **kwargs):  return True

        # If close was called from __del__, we need to wrap all calls such that
        # exceptions are ignored (as there are no guarantees anything still
        # exists outside of this class, e.g. at interpreter exit)
        handler = (suppress if origin == '__del__' else nullcontext)()
        with handler:
            self.debug(f'Called close from: {origin}')

        # Signal threads/workers to exit
        if ('_exit_flag' in self.__dict__) and not self._exit:
            # In certain situations, Event.set can deadlock
            #  (see https://stackoverflow.com/a/73341335/22210498)
            with handler: self._exit_flag.set()

        # Close background thread managers
        for key in ['_block_tasks', '_batch_tasks']:
            if key in self.__dict__:
                with handler: self.__dict__.pop(key).close()

        # Exhaust queue if this is the main process
        if hasattr(self, '_queue') and not hasattr(self, '_ppid'):

            # Discard any queue items if we need to close immediately
            if timeout == 0:
                self._queue.cancel_join_thread()
            else:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except:
                        break

        # Delete cached attributes
        for key in ['_generator', '_queue']:
            with handler: self.__dict__.pop(key, None)

        # Clean up background processes
        for job in self.__dict__.pop('_processes', []):
            pid = code = 'Unknown'
            with handler:
                pid = job.pid
            with handler:
                code = job.exitcode
            with handler:
                self.debug(f'Joining pid={pid} (code={code})')
            with handler:

                # Wait a maximum of `timeout` seconds to join
                try:
                    if timeout:
                        job.join(timeout) or job.close()
                    else:
                        job.terminate()
                except ValueError:
                    try:
                        job.terminate()
                    except:
                        pass
                    self.warning(f'Process {pid} needed to be terminated')

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
        if hasattr(self, '_pidx'):
            return f'Process {self._pidx} (pid={os.getpid()})'
        return f'Main process (pid={os.getpid()})'

    # Internal functions: shouldn't need to be directly accessed by users
    # ===================================================================
    # 
    # High level flow of batch creation:
    # ----------------------------------
    # Batcher.__iter__ -> 
    #   generator ->
    #     _iter_process_batches ->
    #       _queue_batches (if workers > 0)
    #       _generator ->
    #         _generate_batches ->
    #           _blocker ->
    #             Blockset.find_matches
    #             _batcher ->
    #               Blockset._parse
    #               _extract_features
    #               _finalize_batch
    #               yield batch

    def _iter_process_batches(self):
        """ Helper which unifies interface for single/multiple processes """
        try:
            # If using multiprocessing, yield batches from the queue
            if self.workers > 0:
                error = self.error
                debug = self.debug

                def alive(job, verbose=True):
                    if not job.is_alive():
                        if verbose:
                            log = error if job.exitcode else debug
                            log(f'Worker {job.pid} exitcode: {job.exitcode}')
                        return False
                    return True

                # Poll the queue for new batches until all workers exit
                jobs = list(self._processes)
                while not self._exit and any(alive(j, False) for j in jobs):
                    try:
                        yield self._queue.get(timeout=0.1)
                    except Empty:
                        pass
                    # while not self._exit and self._queue.poll(timeout=0.2):
                    #     yield self._queue.recv()
                    jobs = list(filter(alive, jobs))

                # Exit code 3221225477 is STATUS_ACCESS_VIOLATION, which is
                # commonly caused by an issue with data cached on disk. It can
                # also occur when the system runs out of memory, however.
                list(map(alive, jobs))

            # Otherwise, just yield from the threaded generator
            else:
                # However, using 'yield from' will result in _generator
                # being prematurely closed, as python will recursively
                # close generators when they are garbage collected; i.e.:
                # - Batcher.generator() reference is garbage collected
                # - so the 'yield from _generator()' is then closed
                # - and so _generator itself is then closed
                # Further discussion: https://stackoverflow.com/a/74923483
                for batch in self._generator: yield batch

        except KeyboardInterrupt: 
            self.info('KeyboardInterrupt')
            self.close(timeout=0, origin='KeyboardInterrupt') 
            raise
        except Exception as e:    self.error(f'Exception: {e}\n{traceback.format_exc()}')

        # Clean up resources once all batches have been yielded
        # Do not wrap in 'finally', as this will also prematurely
        # close the _generator object when Batcher.generator is
        # garbage collected.
        self.close(origin='Batcher._iter_process_batches')

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
            self.info(f'{self.process_name} Batcher generator starting')
            process = psutil.Process()
            start_t = time.time()
            psutil.cpu_percent()

            # Ensure the lock and flags are created before starting threads
            self._remainder_lock
            if getattr(self, '_pidx', 0) == 0:
                self._first_done.clear()
                self._exit_flag.clear()

            if isinstance(self.dataset, list) and isinstance(self.dataset[0], Dataset):
                dataset = self.dataset[getattr(self, '_pidx', 0)]
                if getattr(self, '_pidx', 0) == 1: self.continuous = False
            else: dataset = self.dataset

            # Samples is a list of dask.Delayed objects or a crest Dataset
            if hasattr(dataset, 'generate_samples'):
                self.debug('Generating dataset blocks...')
                with Stopwatch(logger=self.info) as timer:
                    blocks = dataset.generate_samples(**{
                        'blocksize' : self.blocksize,
                        'numblocks' : self.numblocks,
                        'compute'   : False, 
                        'verbose'   : True, 
                        'logger'    : self._logger,
                        'shuffle'   : self.shuffle,
                    })
                    timer.message = f'Generated {len(blocks)} dataset blocks'
            else: blocks = dataset

            # Verify there are enough sample blocks if we're not duplicating
            if (len(blocks) < self.workers) and (not self.duplicate):
                self.warning('Not enough sample blocks for workers! ' +
                             'Set Batcher.duplicate=True.')
                if getattr(self, '_pidx', 0) >= len(blocks): return

            # Create threaded task executors for generating blocks and batches
            kwargs = {
                'threads'  : min(len(blocks), self.threads),
                'capacity' : min(len(blocks), self.threads) * 2,
                'exitflag' : (lambda: self._exit),
                'exitset'  : self._exit_flag.set,
                'logger'   : self._logger,
            }

            self._batch_tasks = ThreadedFunction(self._batcher, **kwargs)
            self._block_tasks = ThreadedFunction(self._blocker, **kwargs)

            divider = ''.join(['-'] * 57)
            message = '\n\t'.join(['', divider, 'Completed epoch'])

            # If requested, yield batches indefinitely
            while not self._exit:
                with Stopwatch(message, self.info, silent=True) as epoch_log:
                    yield from self._generate_batches(blocks)

                    # Only log a completed epoch if one has actually finished
                    epoch_log.silent = self._exit

                # Break the infinite loop if we're not repeating
                # We don't want to set the exit flag here, as that would stop
                #  all of our background process generators, not just this one
                if not self.repeat: break
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
        subsets = list(blocks)
        start_t = time.time()

        if self.shuffle:
            self.random.shuffle(subsets)

        # If multiprocessing, use only a subset of the
        # overall blocks unless duplicate was set to True
        if hasattr(self, '_pidx') and not self.duplicate:
            subsets = subsets[self._pidx::self.workers]

        # Initialize a new container for leftover samples
        self._remainder = []
        n_block = len(subsets)

        # Send off first block ASAP to minimize time to first batch
        if not self._first and (getattr(self, '_pidx', 0) == 0):
            first, *subsets = subsets
            self._block_tasks([first])

        # Wait for first batch job to signal completion
        if not self._first:
            self.info('Waiting for first set to be completed...')

            if getattr(self, '_pidx', 0) == 0:
                if self._block_tasks.threads < 1:
                    next(self._block_tasks)
                if self._batch_tasks.threads < 1:
                    while not len(self._batch_tasks):
                        time.sleep(WAIT_TIME)
                    yield from self._batch_tasks

            with Stopwatch('Finished waiting on first set', self.info):
                while not (self._first or self._exit):
                    time.sleep(WAIT_TIME)

        # Divide them into small subsets to operate over in parallel
        if len(subsets):
            pertask = min(len(subsets), self._subset_blocks.value)
            n_split = max(1, len(subsets) // pertask)
            subsets = np.array_split(subsets, n_split)
            self.info(f'Using {pertask} block/task, {len(subsets)} task total')

        runtime = lambda r: (time.time() - start_t) / (n_block - r)
        eta_est = lambda r: Stopwatch.readable(runtime(r) * r, 'time')
        running = lambda: self._block_tasks or self._batch_tasks or subsets
        blk_lim = getattr(self.dataset, '__len__', lambda: 10)() * 6

        # Keep executing until all blocks and batches are processed
        while running() and not self._exit:

            # Fill the queue with data that still needs to be processed
            while subsets and not (self._exit or self._block_tasks.is_full()):
                self._block_tasks(subsets.pop(0))

            # Yield any batches that have been generated
            for task in self._batch_tasks:
                yield from getattr(task, 'result', lambda: task)()

            # Clear out any completed block tasks (as no value is returned)
            if self._block_tasks.threads > 0:
                if len(list(self._block_tasks)):
                    left = len(subsets) + len(self._block_tasks)
                    tps = runtime(left)
                    rate = Stopwatch.readable(tps, 'time')
                    eta = Stopwatch.readable(tps * left, 'time')
                    self.info(f'{left} tasks left @ {rate}/task -> ~{eta}\n')

                    # Verify Block objects are being cleared from memory
                    if len(Block._refs) > blk_lim:
                        blk_lim = len(Block._refs)
                        self.warning(f'Blocks may not be clearing memory: ' +
                                     f'{blk_lim} block objects held')
            elif len(self._batch_tasks) == 0:
                next(self._block_tasks)

            # Brief sleep to release GIL while waiting for threads
            time.sleep(WAIT_TIME)

        if running(): self.debug('Exiting _generate_batches early (exit flag)')

        # Yield any remaining samples as the final batch
        if not self._exit and len(self._remainder):
            yield self._finalize_batch(self._remainder)

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
        # Combine multiple blocks into a single array
        self.debug(f'Starting compute of find_matches for {len(blocks)} blocks')
        with Stopwatch(f'Computed {len(blocks)} blocks', self.info) as timer:
            # Need to find a better way to do this
            compute = lambda blocks: da.hstack(da.compute(*blocks))
            try:
                samples = compute([b(self.task_bytes) for b in blocks])
            except TypeError:
                samples = compute(blocks)

            # Log infomation about the samples that were just computed
            blocks = None
            ntotal = Stopwatch.readable(samples.size)
            nbytes = Stopwatch.readable(samples.nbytes, 'byte')
            nblock = samples.blocks.size
            timer.message += f' -> {ntotal} samples ({nblock} blocks, {nbytes})'

        # Exit early if signaled to do so
        if not self._exit:

            # Ensure the _first_done flag is set, and Blocks are cleaned up
            if not samples.size:
                samples = None
                with Stopwatch(f'GC: {len(Block._refs)}', self.debug) as timer:
                    gc.collect()
                    timer.message += f' -> {len(Block._refs)} Block references'
                if not self._first: self._first_done.set()
                return

            # Chunk to more reasonable sizes if current chunks are too small
            # if samples.blocks.size < 10:
            #     samples = samples.rechunk((self.batch_size,))

            # Combine multiple blocks together per task to increase throughput
            num_blocks = samples.blocks.size
            block_bytes = samples.nbytes / num_blocks
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
            self._subset_blocks.value = int(max(1, task_blocks // num_blocks))

            task_blocks = int(max(1, min(num_blocks, task_blocks)))
            task_count = ceil(num_blocks / task_blocks)
            self.info(f'Adding {task_count} batch tasks: ' +
                      f'{task_blocks} blocks/task, ' +
                      f'~{Stopwatch.readable(block_bytes, "byte")}/block')

            # If requested, shuffle order in which sample blocks are computed
            order = np.arange(samples.blocks.size)
            if self.shuffle: self.random.shuffle(order)
            order = iter(order)
        else:
            task_count = 0

        with Stopwatch(f'Queued {task_count} batch tasks', self.info) as timer:

            # Extract batches from blocks in the sample array
            while not self._exit and (idx := list(islice(order, task_blocks))):
                self._batch_tasks(samples.blocks[idx], continuous=self.continuous)

                # Wait until the first batch is done, or signaled to exit
                while not ((self._first or self._block_tasks.threads < 1) or self._exit):
                    time.sleep(WAIT_TIME)
                
                if self.continuous: 
                    # Manually garbage collect Block object as soon as possible
                    samples = None 
                    with Stopwatch(f'GC: {len(Block._refs)}', self.debug) as timer:
                        gc.collect()
                        timer.message += f' -> {len(Block._refs)} Block references'
                    break

            # Close the background thread manager if exit has been signaled
            if self._exit:
                try:
                    self._block_tasks.close()
                except:
                    pass
                timer.silent = True

    def _batcher(self, samples: da.Array) -> list:
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
        # Compute the current samples and extract features if requested
        with Stopwatch(f'Compute/extract {len(samples):,} samples', self.debug):
            samples = self._extract_features(samples.compute())

        if self._exit:
            self.debug(f'Exiting _batcher early due to exit flag')
            return

        n_batch = len(samples) // self.batch_size
        with Stopwatch(f'Generated {n_batch} batches', self.info):
            if self.shuffle: self.random.shuffle(samples)

            # Faster, but discards any (samples % batch_size) samples
            if self.continuous:
                splits  = np.arange(self.batch_size, len(samples), self.batch_size)
                list(map(self._queue.put, map(self._finalize_batch, np.array_split(samples, splits))))
                samples = []
            else:
                # Use a lock to ensure _remainder is handled by only one thread
                with self._remainder_lock:
                    if 0 < len(self._remainder) < self.batch_size:
                        samples = np.append(self._remainder, samples, 0)

                    splits  = np.arange(self.batch_size, len(samples), self.batch_size)
                    samples = np.array_split(samples, splits)
                    if len(samples[-1]) < self.batch_size:
                        *samples, self._remainder = samples
                    else: self._remainder = []
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

            # Manually garbage collect Block object as soon as possible
            with Stopwatch(f'GC: {len(Block._refs)}', self.debug) as timer:
                gc.collect()
                timer.message += f' -> {len(Block._refs)} Block references'
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
            if len(features) and isinstance(features[0], list):
                return list(map(partial(_extract, sample), features))
            return sample.to_list(features)

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

        def _parse(batch, features):
            """ Follow the nesting structure given by the features """
            if len(features) and isinstance(features[0], list):
                return list(map(_parse, zip(*batch), features))
            return dict(zip(features, map(np.array, zip(*batch))))

        return batch if self.features is None else _parse(batch, self.features)

    def _queue_batches(self,
                       process_ix: int,
                       parent_pid: int,
                       queue: mp.queues.Queue,
                       exit_flag: mp.synchronize.Event,
                       first_done: mp.synchronize.Event,
                       sub_blocks: mp.sharedctypes.Synchronized,
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
        try:
            from terrahydro.data.utils.UploadMonitor import UploadMonitor
        except:
            pass

        # Make sure we catch and log any exceptions, as they will disappear
        #  silently otherwise (since we're in a background process here)
        try:
            # Ensure we are properly catching any segmentation faults
            faulthandler.enable()

            # Store the process metadata and shared objects
            self.__dict__.update({
                '_pidx': process_ix,
                '_ppid': parent_pid,

                '_exit_flag'     : exit_flag,
                '_first_done'    : first_done,
                '_subset_blocks' : sub_blocks,
                '_queue'         : queue,
            })

            # When duplicate is set, generate new randomness per process
            if self.duplicate: self.random = np.random.default_rng(process_ix)
            self.debug(f'{self.process_name} initialized')

            # import memray
            # with memray.Tracker(f'output.bin.{os.getpid()}'):

            # Start adding batches to the queue
            for batch in self._generator:
                # with tf.device('GPU:0'):
                #     batch = [{k:tf.convert_to_tensor(v) for k,v in b.items()} for b in batch]

                # Ensure exit flag is monitored while waiting on full queue
                while not self._exit:
                    try:
                        queue.put_nowait(batch)
                        break
                    except Full:
                        while (not self._exit) and queue.full():
                            time.sleep(WAIT_TIME)

                if self._exit: break

        except Exception as e:
            self.error(f'{self.process_name} exception: {e}\n' +
                       f'{traceback.format_exc()}')
            raise

        # Ensure queued items do not block this process from joining
        finally:
            if self._exit: self.close(0, '_queue_batches')
            self.debug(f'{self.process_name} waiting for queue to clear')
            while len(queue._buffer):
                if not psutil.pid_exists(self._ppid):
                    queue.cancel_join_thread()
                time.sleep(WAIT_TIME)
            self.info(f'{self.process_name} exiting')

    def _is_picklable(self):
        """ Checks that the dataset is picklable before spawning processes """
        try:
            pkl.dumps(self.dataset)
        except pkl.PicklingError:
            message = f'Dataset {self.dataset} is required to be picklable '
            message += 'when using multiprocessing (i.e. Batcher.workers > 0)'
            self.error(message)
            raise Exception(message)

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
        self._exit_flag.clear()

        # Create the spawning context and the queue used to transfer batches
        mp_context = mp.get_context('spawn')
        self._queue = mp_context.Queue(self.max_queue)
        # self._queue, conn = mp_context.Pipe(False)
        shared_attr = (
            os.getpid(),
            self._queue,
            self._exit_flag,
            self._first_done,
            self._subset_blocks,
        )

        # Create and start the background processes
        kwargs = {'target': self._queue_batches, 'daemon': True}
        create = lambda i: mp_context.Process(args=(i,) + shared_attr, **kwargs)
        jobs = [p.start() or p for p in map(create, range(self.workers))]
        assert (len(jobs) == self.workers), jobs

        # Wait a second for them to start, then ensure that they are running
        time.sleep(1)
        if any(not job.is_alive() for job in jobs):
            message = 'Batcher processes are stopping immediately. This is '
            message += 'possibly due to issues pickling the given Dataset, if '
            message += 'no other exceptions are logged.'
            message += f'\nExit codes: {[job.exitcode for job in jobs]}'
            self.error(message)
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
        orphaned = hasattr(self, '_ppid') and not psutil.pid_exists(self._ppid)
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
        return self._first_done.is_set()

    @cached_property
    def _subset_blocks(self) -> mp.sharedctypes.Synchronized:
        """ Synchronized value across processes which defines subset blocks """
        shared_value = mp.get_context('spawn').Value(c_int)
        shared_value.value = 1
        return shared_value

    @cached_property
    def _remainder_lock(self) -> _thread.LockType:
        """ Lock ensuring _remainder is modified by one thread at a time """
        return threading.Lock()

    @cached_property
    def _logger(self):
        """ Create the logging object which writes logs to a file """
        format_key = [
            '%(asctime)23s',
            '%(process)6s',
            '%(threadName)10.10s',
            '%(lineno)3s:%(filename)-9.9s..',
            '%(levelname)7s',
            '%(message)s',
        ]
        log_format = logging.Formatter(' | '.join(format_key))

        logger = logging.getLogger(Path(self.logfile).name)
        if logger.hasHandlers():
            logger.handlers.clear()

        # abseil hijacks the root logger
        logger.propagate = False

        # If a logfile is requested, use a file handler
        if self.logfile is not None:

            # Add a handler for error logs that also prints to sys.stderr
            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            handler.setFormatter(log_format)
            logger.addHandler(handler)

            # A rotating handler would be nice, but doesn't work in Windows
            handler = logging.FileHandler(self.logfile)

            # Define the log header
            replace = {
                'process': 'pid',
                'levelname': 'level',
            }
            pattern = re.compile(r'\((.*?)\)')
            labels = list(chain(*map(pattern.findall, format_key)))
            values = [replace.get(label, label) for label in labels]
            inserts = ' | '.join(format_key) % dict(zip(labels, values))
            headrow = inserts.replace('.. ', '')
            divider = re.sub(r'[^|]', '=', headrow)
            header = f'{headrow}\n{divider}\n'

            # If we're in the main process, delete the prior logfile
            if not mp.current_process().daemon:
                filename = Path(self.logfile).absolute()

                # Backup previous log file if it exists
                if filename.exists():

                    # Remove any existing backups
                    for backup in filename.parent.glob(f'{filename.name}*.backup'):
                        try:
                            backup.unlink()
                        except Exception as e:
                            print(f'Exception removing file {backup}: {e}')

                    # Make backup of main Batcher.log, as well as any process logs
                    for plog in filename.parent.glob(f'{filename.name}.*'):
                        shutil.copy(plog, f'{plog.as_posix()}.backup')
                        try:
                            plog.unlink()
                        except Exception as e:
                            print(f'Exception removing file {plog}: {e}')

                    shutil.copy(filename, Path(f'{filename.as_posix()}.backup'))
                Path(self.logfile).write_text(header)

            # Otherwise, add a second handler specifically for this process
            elif self.workers > 1:
                filename = Path(self.logfile).as_posix() + f'.{os.getpid()}'
                handler2 = logging.FileHandler(filename)
                Path(filename).write_text(header)
                handler2.setFormatter(log_format)
                logger.addHandler(handler2)

        # Otherwise just log to sys.stderr
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(log_format)
        logger.setLevel(self.loglevel)
        logger.addHandler(handler)
        return logger

    def _log(self, method, *args, **kwargs):
        """ Helper which allows modifying the stacklevel for correct labels """
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 2
        getattr(self._logger, method)(*args, **kwargs)   # CC: Test here
