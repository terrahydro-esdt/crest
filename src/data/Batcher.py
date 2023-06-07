from numpy.lib.stride_tricks import as_strided
from collections.abc import Collection
from functools import partial, cached_property
from numbers import Number
from pathlib import Path 
from threading import Timer, Lock, Event
from queue import Empty
from tqdm.auto import tqdm

import multiprocessing as mp
import dask.array as da
import numpy as np 
import traceback
import logging
import time
import os

from crest.src.data.loading import Dataset, StructuredDataset, SampleSet
from crest.src.utils import Stopwatch
from crest.src.base  import BaseAbstract
from .ThreadedFunction import ThreadedFunction



class Batcher:#(BaseAbstract):
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
        [{'a':<batch of 'a' values>, 'b': <batch of 'b' values>}, 
         {'c': <batch of 'c' values>}]. If any features are missing in the
        dataset, an exception is raised.
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
    def __init__(self, 
        dataset    : Dataset | StructuredDataset,
        batch_size : int,
        features   : list  = [],
        workers    : int   = 2,
        threads    : int   = 1,
        shuffle    : bool  = True,
        repeat     : bool  = False,
        duplicate  : bool  = False,
        max_queue  : int   = 100,
        task_bytes : float = 5e7,
        logfile    : str   = 'Batcher.log',
        loglevel   : int   = logging.INFO,
        seed       : int | None = None,
    ):
        self.dataset    = dataset
        self.n_batch    = batch_size 
        self.features   = features
        self.workers    = workers if shuffle else 0
        self.threads    = threads if shuffle else 1
        self.shuffle    = shuffle
        self.repeat     = repeat
        self.duplicate  = duplicate
        self.max_queue  = max_queue
        self.task_bytes = task_bytes
        self.logfile    = logfile
        self.loglevel   = loglevel
        self.random     = np.random.default_rng(seed)


    def __getstate__(self):
        """ Ensure unpickle-able objects are not included """
        return {
            'dataset'    : self.dataset,
            'n_batch'    : self.n_batch,
            'features'   : self.features,
            'workers'    : self.workers,
            'threads'    : self.threads,
            'shuffle'    : self.shuffle,
            'repeat'     : self.repeat,
            'duplicate'  : self.duplicate,
            'max_queue'  : self.max_queue,
            'task_bytes' : self.task_bytes,
            'logfile'    : self.logfile,
            'loglevel'   : self.loglevel,
            'random'     : self.random,
        }


    def __iter__(self): yield from self.generator()
    def __next__(self): return next(self.generator())
    def __exit__(self, *args, **kwargs): self.close()
    def __enter__(self):
        """ Create any necessary resources and start generating batches """      
        if self.workers > 0: self._processes
        else:                self._generator
        return self


    def generator(self, show_timing=False):
        """ Top-level function to generate batches. 

        The only reason this function would be called rather than iterating
        the Batcher object itself, would be if timings should be logged.

        Parameters
        ----------
        show_timing : bool
            Whether timing logs should be printed to show batch iteration speed

        """
        def _batches():
            # If using multiprocessing, yield from the cached queue
            if self.workers > 0:
                while any(job.is_alive() for job in self._processes):
                    try:          yield self._queue.get(timeout=0.1)
                    except Empty: pass
                    except KeyboardInterrupt: break
                    except Exception as e: 
                        self.logger.error(f'Exception: {e}')
                        break 
            else:
                # Otherwise, just yield from the threaded generator
                # Using 'yield from' however will result in _generator being 
                # prematurely closed, as python will recursively close generators
                # when they are garbage collected (i.e. generator() reference is
                # garbage collected, so the 'yield from _generator()' is closed,
                # and so _generator itself is closed). 
                # More discussion here: https://stackoverflow.com/a/74923483
                for batch in self._generator: yield batch

            # Clean up resources once all batches have been yielded
            self.close()

        bar_kwargs = {
            'unit_scale' : True,
            'smoothing'  : 0,
            'disable'    : not show_timing,
        }
        with tqdm(unit=' Batches', **bar_kwargs) as pbar, \
             tqdm(unit=' Samples', **bar_kwargs) as pbar2:
            start = time.time()
            for i, batch in enumerate( _batches() ):
                if show_timing and (i == 0):
                    elapsed = time.time() - start
                    pbar.clear()
                    pbar2.clear()
                    print(f'Time to first batch: {elapsed:.1f} seconds\n')
                    pbar.unpause()
                    pbar2.unpause()
                pbar.update(1)
                pbar2.update(self.n_batch)
                yield batch


    @property
    def running_jobs(self):
        return self.__dict__.get('_processes', [])

    
    def close(self):
        """ Close and delete all thread / process resources """
        if '_exit_flag' in self.__dict__:
            self._exit_flag.set()

        if '_processes' in self.__dict__:
            for job in self._processes:
                job.terminate()

        for key in ['_generator', '_processes', '_queue']:
            if key in self.__dict__:
                del self.__dict__[key]

        for key in ['block', 'batch']:
            task_set = getattr(self, f'_{key}_tasks', None)
            if task_set is not None: task_set.close()


    @cached_property
    def logger(self):
        """ Create the logging object which writes logs to a file """
        message = '%(asctime)s | %(process)6d | %(threadName)s | %(levelname)7s | %(message)s'
        handler = logging.FileHandler(self.logfile)
        handler.setFormatter( logging.Formatter(message) )
        if not mp.current_process().daemon: Path(self.logfile).write_text('')

        logger = logging.getLogger('Batcher')
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.loglevel)
        logger.addHandler(handler)
        return logger


    @cached_property
    def _generator(self):
        """ Perform the initial setup of batch creation, and return the generator """
        try:
            self.logger.info('Starting Batcher._generator')

            # Ensure the lock and flags are created before starting threads
            self._remainder_lock
            self._first_done.clear()
            self._exit_flag.clear()

            # Samples is a list of dask.Delayed objects or a crest Dataset
            samples = self.dataset
            if not isinstance(samples, list):
                samples = self.dataset.generate_samples(compute=False, verbose=False)

            if (len(samples) < self.workers) and (not self.duplicate):
                self.logger.warning(f'Not enough sample blocks for workers! Set duplicate=True.')

            # Create task executors for generating blocks and batches
            kwargs = {
                'threads'  : min(len(samples), self.threads),
                'capacity' : min(len(samples), self.threads) * 2,
                'exitflag' : self._exit_flag,
                'logger'   : self.logger,
            }
            self._batch_tasks = ThreadedFunction(self._batcher, **kwargs)
            self._block_tasks = ThreadedFunction(self._combine, **kwargs)

            # If requested, yield samples indefinitely
            while not self._exit_flag.is_set():
                with Stopwatch('\n\t-------------------\n\tCompleted epoch', self.logger.info):
                    if self.shuffle: self.random.shuffle(samples)   

                    # If multiprocessing, use only a subset of the overall samples                 
                    if hasattr(self, 'i'): subsets = samples[self.i::self.workers]
                    else:                  subsets = list(samples) 

                    # Initialize a new container for leftover samples
                    self._remainder = []

                    # Send off first block ASAP to minimize time to first batch
                    if not self._first_done.is_set():
                        first, *subsets = subsets
                        self._block_tasks([first])

                    # Divide them into small subsets to operate over in parallel
                    if len(subsets):
                        n_block = 1
                        n_split = max(1, len(subsets) // n_block) 
                        subsets = np.array_split(subsets, n_split)

                    # Yield the generated batches
                    yield from self._iter_batches(subsets)

                # Break the infinite loop if we're not repeating
                if not self.repeat: break

        except (KeyboardInterrupt, Exception) as e:   
            msg = f'\nException: {e}\n{traceback.format_exc()}'
            print(msg)
            self.logger.error(msg)
            print(f'\nReceived exit signal; stopping threads within 3 seconds...')
            Timer(3, lambda: os._exit(0)).start() 


    def _iter_batches(self, subsets):
        """ Yield batches of samples """
        block_tasks = self._block_tasks
        batch_tasks = self._batch_tasks

        # Wait for first batch job to signal completion
        while not self._first_done.is_set(): time.sleep(0.01)

        # Keep executing until all blocks and batches are processed
        while (block_tasks or batch_tasks or subsets) and not self._exit_flag.is_set():

            # Send off any remaining tasks
            while len(subsets) and not block_tasks.is_full():
                block_tasks( subsets.pop(0) )

            # Yield any available batches
            for task in batch_tasks:
                yield from task.result()

            # Clear any completed block tasks
            # list(block_tasks)
            [b.result() for b in block_tasks]

            # Brief sleep to release GIL while waiting for threads
            time.sleep(0.01)

        # Yield any remaining samples
        if len(self._remainder): 
            yield self._to_dict(self._remainder) 


    def _combine(self, blocks: Collection) -> None:
        """ Combine the given dataset blocks into a single dask Array,
            which is then shuffled and subdivided, and then sent on to
            the _batcher threads for final processing.
        
        Parameters
        ----------
        blocks    : Collection
            The collection of dask.delayed dataset blocks that should
            be computed to gather the dask.Array[Sample] objects.

        """
        # Combine multiple blocks into a single array
        self.logger.debug(f'Starting compute of {len(blocks)} block(s)...')
        with Stopwatch(f'Computed {len(blocks)} block(s)', self.logger.debug):
            samples = da.hstack( da.compute(*blocks) )
            message = f'Computed samples from block(s) (shape='
            message+= Stopwatch.readable(samples.size, units='size')
            message+= f'  bytes={Stopwatch.readable(samples.nbytes)}'
            message+= f'  chunks={samples.chunksize})'
            self.logger.debug(message)

        # Chunk to more reasonable sizes if current sizes are too small
        if samples.chunksize[0] < 10:
            samples = samples.rechunk((self.n_batch,))

        # Extract batches from each block in the array in parallel
        order = np.arange(samples.numblocks[0])
        if self.shuffle: self.random.shuffle(order)

        # Send off the first block ASAP to minimize time to first batch
        first, *order = order
        self._batch_tasks(samples.blocks[first])
        while not self._first_done.is_set(): time.sleep(0.01)        

        # Process up to 100MB at once to increase throughput,
        # and improve randomness by shuffling across blocks
        n_bytes = samples.nbytes / samples.numblocks[0]
        n_block = int(min(samples.numblocks[0], max(1, self.task_bytes // n_bytes)))
        extract = lambda i: order[i::n_block]
        chunks  = map(list, zip(*map(extract, range(n_block))))

        message = f'Adding {samples.blocks.size//n_block} batch tasks, '
        message+= f'with {n_block} blocks/task, ~{Stopwatch.readable(n_bytes)}'
        message+= f'/block ({Stopwatch.readable(n_block*n_bytes)})'
        self.logger.info(message)
        list(map(self._batch_tasks, map(samples.blocks.__getitem__, chunks)))

        # Ensure any remaining blocks are also processed
        remain = len(order) % n_block
        if remain: self._batch_tasks(samples.blocks[order[-remain:]])


    def _batcher(self, samples: da.Array) -> list:
        """ Separate the given sample array into batches, and transform
            Sample objects into feature dictionaries if requested.

            Parameters
            ----------
            samples : da.Array
                dask.Array object containing the Samples.

            Returns
            -------
            list
                List of batches, where each batch is either a list of 
                Sample objects (when no features are requested); or a
                (possibly nested) list of feature dictionaries. 

        """
        # Compute the current samples and extract features if requested
        with Stopwatch(f'Compute+feature extraction {len(samples)} samples', self.logger.debug):
            samples = self._extract_features( samples.compute() )

        with Stopwatch(f'Generated {len(samples)//self.n_batch} batches', self.logger.debug):
            if self.shuffle: self.random.shuffle(samples)

            # Use a lock to ensure _remainder is handled by only one thread
            with self._remainder_lock:
                if 0 < len(self._remainder) < self.n_batch:
                    samples = np.append(self._remainder, samples, 0)

                # Calculate attributes to create a view of the samples
                n_group = len(samples) // self.n_batch
                i_size  = samples.itemsize
                shape   = (n_group, self.n_batch) + samples.shape[1:]
                strides = (self.n_batch*i_size, i_size) + samples.strides[1:]

                # Save any remainder samples
                self._remainder = samples[n_group * self.n_batch:] 
            
            # Create a strided view and create the batch feature dicts 
            strided = as_strided(samples, shape=shape, strides=strides)
            parsed  = list(map(self._to_dict, strided))

            # Signal that the first batch set is now completed
            if not self._first_done.is_set(): self._first_done.set()
            return parsed 


    def _extract_features(self, samples: SampleSet) -> np.ndarray:
        """ Extract the requested features from the Sample objects """
        def _extract(features, sample):
            """ Nest the data in the same manner as features """
            if isinstance(features[0], list):
                return list(map(partial(_extract, sample=sample), features))
            return sample.to_list(features)
        np_extract = np.frompyfunc(partial(_extract, self.features), nin=1, nout=1)
        return np_extract(samples) if len(self.features) else samples[:]


    def _to_dict(self, batch: np.ndarray) -> dict[str, np.ndarray] | np.ndarray:
        """ Transform the batch array into the final feature dictionary """
        def _parse(features, batch):
            """ Follow the nesting structure given by the features """
            if isinstance(features[0], list):
                return list(map(_parse, features, zip(*batch)))
            return dict(zip(features, np.array(list(zip(*batch)))))
        return _parse(self.features, batch) if len(self.features) else batch


    def _put(self, queue, i):
        """ Helper to put batches into a multiprocessing queue.

        Parameters
        ----------
        queue
            The multiprocessing Queue object to put batches into. The main
            process will monitor this queue to receive batches and yield 
            them to the rest of the program.
        i
            Integer specifying which worker process this is (ranging from
            0 to `workers-1`). This is used to select which subset of dataset
            blocks to operate on when duplicate is set to False.

        """
        if self.duplicate: self.random = np.random.default_rng(i)
        else:              self.i = i
        list(map(queue.put, self._generator))


    @cached_property
    def _exit_flag(self):
        """ Flag to signal Batcher exit """
        return Event()


    @cached_property
    def _first_done(self):
        """ Flag to signal the first batch has been generated """
        return Event()


    @cached_property
    def _remainder_lock(self):
        """ Lock ensuring _remainder is modified by one thread at a time """
        return Lock()


    @cached_property
    def _processes(self):
        """ Create worker background processes """
        ctx    = mp.get_context('spawn')
        queue  = self._queue = ctx.Queue(self.max_queue)
        kwargs = {
            'target' : self._put,
            'daemon' : True,
        }
        # Note that mp.Pool + mp.Manager is a fair bit slower than using Process
        self.logger.info(f'Starting {self.workers} background processes')
        jobs = [ctx.Process(args=(queue, i), **kwargs) for i in range(self.workers)]
        [job.start() for job in jobs]
        return jobs