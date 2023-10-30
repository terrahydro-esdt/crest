from numpy.lib.stride_tricks import as_strided
from collections.abc import Collection
from functools import partial, cached_property
from itertools import chain
from numbers import Number
from pathlib import Path
from queue import Empty
from tqdm.auto import tqdm
from dask.delayed import Delayed 

import logging
import multiprocessing as mp
import threading, _thread
import dask.array as da
import numpy as np
import traceback
import time

from crest.data.loading.Dataset import Dataset
from crest.data.loading.StructuredDataset import StructuredDataset
from crest.data.loading.SampleSet import SampleSet

from crest.utils.Stopwatch import Stopwatch
from crest.data.ThreadedFunction import ThreadedFunction


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
            [{'a':<batch of 'a' values>, 'b': <batch of 'b' values>},
             {'c': <batch of 'c' values>}]. If any features are missing in the
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
        features   : list | None = None,
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


    def __del__(self):  self.close(origin='__del__')
    def __iter__(self): yield from self.generator()
    def __next__(self): return next(self.generator())
    def __exit__(self, *args, **kwargs): self.close(origin='__exit__\n')
    def __enter__(self):
        """ Create any necessary resources and start generating batches """
        if self.workers > 0: self._processes
        else:                self._generator
        return self


    def   debug(self, message): self._logger.debug(message)
    def    info(self, message): self._logger.info(message)
    def warning(self, message): self._logger.warning(message)
    def   error(self, message): self._logger.error(message);\
        print(message)


    def close(self, timeout: Number = 10, origin=''):
        """ Close and delete all thread / process resources """
        # Need to wrap _everything_ in try/except since we may be in __del__
        try: self.info('Called close'+(f' from {origin}' if origin else ''))
        except: pass

        if '_exit_flag' in self.__dict__:
            try:    self._exit_flag.set()
            except: pass

        if '_processes' in self.__dict__:
            for job in self._processes:
                try:    pid  = job.pid 
                except: pid  = 'UNKNOWN' 
                try:    code = job.exitcode
                except: code = 'UNKNOWN'
                try: 
                    # Wait a maximum of `timeout` seconds to join
                    try: self.debug(f'Joining pid={pid} (code={code})...')
                    except: pass
                    job.join(timeout)
                    try: self.debug(f'Process {pid} was joined successfully')
                    except: pass
                except:
                    try: self.debug(f'Process {pid} needed to be terminated')
                    except: pass
                finally: 
                    try: job.terminate()
                    except: pass
    
        for key in ['_generator', '_processes', '_queue']:
            if key in self.__dict__:
                try:
                    del self.__dict__[key]
                except: pass

        for key in ['block', 'batch']:
            try:    
                task_set = getattr(self, f'_{key}_tasks', None)
                if task_set is not None: task_set.close()
            except: pass
            

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


    def generator(self, show_timing=False):
        """Top-level function to generate batches.

        The only reason this function would be called rather than iterating
        the Batcher object itself, would be if timings should be logged.

        Parameters
        ----------
        show_timing : bool
            Whether timing logs should be printed to show batch iteration speed

        """
        def _batches():
            try:
                # If using multiprocessing, yield batches from the queue
                if self.workers > 0:
                    alive = lambda: [job.is_alive() for job in self._processes]
                    codes = lambda: [job.exitcode   for job in self._processes]
                    while any( alive() ):
                        try:          yield self._queue.get(timeout=0.1)
                        except Empty: pass
                    
                    # Exit code 3221225477 is STATUS_ACCESS_VIOLATION, which 
                    # appears to be caused by an issue with data cached on 
                    # disk. If using a data cache, see if clearing fixes it.
                    self.debug(f'Workers alive: {alive()} (codes: {codes()})')


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

            except KeyboardInterrupt: self.info('KeyboardInterrupt')
            except Exception as e:    self.error(f'Exception: {e}')

            # Clean up resources once all batches have been yielded
            # Do not wrap in 'finally', as this will also prematurely
            # close the _generator object when Batcher.generator is
            # garbage collected.
            self.close(origin='generator._batches')

        # Display tqdm progress statistics if requested
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
                    pbar.clear();   pbar2.clear()
                    print(f'Time to first batch: {elapsed:.1f} seconds\n')
                    pbar.unpause(); pbar2.unpause()
                pbar.update(1);     pbar2.update(self.n_batch)
                yield batch



    # Internal functions: shouldn't need to be directly accessed by users
    # ===================================================================

    @cached_property
    def _generator(self):
        """Perform setup and generation of batches.

        Notes
        -----
        This property, when called, starts the actual generation of batches.
        If it has already been called (and the Batcher has not been closed),
        this property returns the already running batch generator.

        """
        def generator_loop(blocks: list):
            """ Core generator loop which yields batches """
            if self.shuffle: self.random.shuffle(blocks)

            # If multiprocessing, use only a subset of the overall blocks
            if hasattr(self, '_pidx'):
                subsets = blocks[self._pidx::self.workers]
            else: subsets = list(blocks)

            # Initialize a new container for leftover samples
            self._remainder = []

            # Send off first block ASAP to minimize time to first batch
            if not self._first_done.is_set() and getattr(self, '_pidx', 0)==0:
                first, *subsets = subsets
                self._block_tasks([first])

            # Divide them into small subsets to operate over in parallel
            if len(subsets):
                n_block = 1
                n_split = max(1, len(subsets) // n_block)
                subsets = np.array_split(subsets, n_split)

            # Wait for first batch job to signal completion
            self.debug('Waiting for first set to be completed...')
            while not self._first_done.is_set():
                if self._exit_flag.is_set(): return
                time.sleep(0.01)
            self.debug('Finished waiting; first set is now complete')

            # Keep executing until all blocks and batches are processed
            running = lambda: self._block_tasks or self._batch_tasks or subsets
            while running() and not self._exit_flag.is_set():

                # Fill the queue with data that still needs to be processed
                while len(subsets) and not self._block_tasks.is_full():
                    self._block_tasks( subsets.pop(0) )
                    
                    if not len(subsets) or self._block_tasks.is_full():
                        self.debug(f'Queued blocks ({len(subsets)} remaining)')

                # Yield any batches that have been generated
                for task in self._batch_tasks: yield from task.result()

                # Clear out any completed block tasks (as no value is returned)
                list(self._block_tasks)

                # Brief sleep to release GIL while waiting for threads
                time.sleep(0.01)

            if running(): 
                self.debug('Exiting generator_loop while due to exit flag')

            # Yield any remaining samples as the final batch
            if not self._exit_flag.is_set() and len(self._remainder):
                yield self._to_dict(self._remainder)

        try:
            self.info('Starting Batcher._generator')

            # Ensure the lock and flags are created before starting threads
            self._remainder_lock
            self._first_done.clear()
            self._exit_flag.clear()

            # Samples is a list of dask.Delayed objects or a crest Dataset
            if hasattr(self.dataset, 'generate_samples'):
                with Stopwatch('Generated dataset blocks in', self.debug):
                    blocks = self.dataset.generate_samples(6e8, **{
                        'compute' : False, 
                        'verbose' : True, 
                        'logger'  : self.info,
                    })
            else: blocks = self.dataset

            # Verify there are enough sample blocks if we're not duplicating
            if (len(blocks) < self.workers) and (not self.duplicate):
                message = 'Not enough sample blocks for workers! '
                message+= 'Set Batcher.duplicate=True.'
                self.error(message)
                if getattr(self, '_pidx', 0) >= len(blocks): return

            # Create threaded task executors for generating blocks and batches
            kwargs = {
                'threads'  : min(len(blocks), self.threads),
                'capacity' : min(len(blocks), self.threads) * 2,
                'exitflag' : self._exit_flag,
                'logger'   : self._logger,
            }
            self._batch_tasks = ThreadedFunction(self._batcher, **kwargs)
            self._block_tasks = ThreadedFunction(self._combine, **kwargs)

            divider = ''.join(['-']*19)
            message = '\n\t'.join(['', divider, 'Completed epoch'])

            # If requested, yield batches indefinitely
            while not self._exit_flag.is_set():
                with Stopwatch(message, self.info, silent=True) as epoch_log:
                    yield from generator_loop(list(blocks))

                    # Only log a completed epoch if one has actually finished
                    epoch_log.silent = self._exit_flag.is_set()

                # Break the infinite loop if we're not repeating
                # We don't want to set the exit flag here, as that would stop
                #  all of our background process generators, not just this one
                if not self.repeat: break
            else: self.debug('Exiting _generator while loop due to exit flag')

        except (KeyboardInterrupt, Exception) as e:
            message = f'\nException: {e}\n{traceback.format_exc()}\n'
            # message+= f'Forcing halt in 3 seconds...'
            self.error(message)
            self.close(origin='_generator exception')
            # threading.Timer(3, lambda: os._exit(0)).start()


    def _combine(self, blocks: Collection) -> None:
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
        self.debug(f'Starting compute of {len(blocks)} block(s)...')
        with Stopwatch(f'Computed {len(blocks)} block(s)', self.debug):
            with Stopwatch('gen block', self.debug):
                blocks  = [b if isinstance(b, Delayed) else b() for b in blocks]
            with Stopwatch('computation', self.debug):
                samples = da.compute(*blocks)
            self.info(f'Samples: {type(samples)}')
            try: self.info(f'Shape: {len(samples)} {type(samples[0])} {samples[0].shape}')
            except: pass
            with Stopwatch('hstack', self.debug):
                samples = da.hstack(samples)# da.compute(*blocks) )

            message = f'Computed samples from block(s) (size='
            message+= Stopwatch.readable(samples.size, units='size')
            message+= f'  bytes={Stopwatch.readable(samples.nbytes)}'
            message+= f'  chunks={samples.chunksize})'
            self.debug(message)

        # Double check exit flag
        if self._exit_flag.is_set(): 
            self._block_tasks.close()
            return 

        # Ensure the _first_done flag is set before returning
        if not samples.size:
            if not self._first_done.is_set(): self._first_done.set()
            return

        # Chunk to more reasonable sizes if current chunks are too small
        if samples.chunksize[0] < 10:
            samples = samples.rechunk((self.n_batch,))

        # Extract batches from each block in the array in parallel
        order = np.arange(samples.numblocks[0])
        if self.shuffle: self.random.shuffle(order)

        # Send off the first block ASAP to minimize time to first batch
        first, *order = order
        self._batch_tasks(samples.blocks[first])
        while not self._first_done.is_set():
            time.sleep(0.01)
            if self._exit_flag.is_set(): return

        # Process up to 100MB at once to increase throughput,
        # and improve randomness by shuffling across blocks
        n_bytes = samples.nbytes / samples.numblocks[0]
        n_block = max(1, self.task_bytes // n_bytes)
        n_block = int(min(samples.numblocks[0], n_block))
        extract = lambda i: order[i::n_block]
        chunks  = map(list, zip(*map(extract, range(n_block))))

        message = f'Adding {samples.blocks.size//n_block} batch tasks, '
        message+= f'with {n_block} blocks/task, ~{Stopwatch.readable(n_bytes)}'
        message+= f'/block ({Stopwatch.readable(n_block*n_bytes)})'
        self.info(message)

        remain = len(order) % n_block 
        tasks  = map(samples.blocks.__getitem__, chunks)

        # Ensure any remaining blocks are also processed
        if remain: 
            final = (samples.blocks[order[-remain:]] for _ in range(1))
            tasks = chain(tasks, final)

        for task in tasks:
            if self._exit_flag.is_set(): break
            self._batch_tasks(task)

        if self._exit_flag.is_set():
            self._block_tasks.close()


    def _batcher(self, samples: da.Array) -> list:
        """Step 2: Separate the given sample array into batches.

        Notes
        -----
        This function implements the second of two steps that the Batcher
        performs when generating batches, and is operated in parallel by
        however many threads were requested by Batcher.threads.

        This function is given a dask array of (uncomputed) Samples, created
        by the first step (Batcher._combine). This array is:
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
        with Stopwatch(f'Compute/extract {len(samples)} samples', self.debug):
            samples = self._extract_features( samples.compute() )

        n_batch = len(samples) // self.n_batch
        with Stopwatch(f'Generated {n_batch} batches', self.debug):
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


    def _extract_features(self, samples: SampleSet | np.ndarray) -> np.ndarray:
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

        def _extract(sample, features=self.features):
            """ Follow the nesting structure given by the features """
            if len(features) and isinstance(features[0], list):
                return list(map(partial(_extract, sample), features))
            return sample.to_list(features)
        return np.frompyfunc(_extract, nin=1, nout=1)(samples)


    def _to_dict(self, batch: np.ndarray) -> np.ndarray | dict | list:
        """Step 2.2: Transform the batch array into the final feature dict.

        Notes
        -----
        This is step two in the _batcher function, where the batch of Sample
        objects are converted into the requested features dictionaries (based
        on the value of Batcher.features).

        Parameters
        ----------
        batch : np.ndarray
            The batch of Sample objects to extract from.

        Returns
        -------
        np.ndarray | dict | list
            The return type depends on what value the `features` parameter
            was given when the Batcher object was created:
            - if nothing was given, Batcher.features=None, and so the batch
              is returned unmodified (and so will be a numpy array of Samples)
            - if a list of strings was given for `features`, the return value
              will be a dictionary in which the keys are the requested features
              and the values are the numpy arrays shaped [batch size, ...] for
              each requested feature
            - if a nested list of strings was given (e.g. [['f1'], ['f2']]),
              the the returned object will be a nested list of dictionaries
              with the same structure as the requested features (e.g.
              [{'f1': <np.ndarray>}, {'f2': <np.ndarray>}])

        """
        def _parse(batch, features=self.features):
            """ Follow the nesting structure given by the features """
            if len(features) and isinstance(features[0], list):
                return list(map(_parse, zip(*batch), features))
            return dict(zip(features, map(np.array, zip(*batch))))
        return batch if self.features is None else _parse(batch)


    def _put(self,
        queue     : mp.queues.Queue,
        proc_idx  : int,
        exitflag  : mp.synchronize.Event,
        firstdone : mp.synchronize.Event,
    ) -> None:
        """Helper to put batches into a multiprocessing queue.

        Notes
        -----
        This function is the entry point for worker processes.

        Parameters
        ----------
        queue    : multiprocessing.queues.Queue
            The multiprocessing Queue object that workers will put batches
            into. The main process will monitor this queue to receive batches
            and yield them to the rest of the program.
        proc_idx : int
            Integer specifying which worker process this is (ranging from
            0 to `workers-1`). This is used to select which subset of dataset
            blocks to operate on when Batcher.duplicate is set to False.
        exitflag : multiprocessing.synchronize.Event
            Event used to signal the process to exit.

        """
        self.info(f'Starting process {proc_idx}')

        # UploadMonitor is required to import in order to register it
        # Since this method is the entry point for new processes, it's the
        #   easiest spot to import and register the class. It doesn't really
        #   make sense to import it here from an organizational standpoint
        #   though, so it does need to be moved somewhere else eventually.
        try:  from terrahydro.src.data.utils.UploadMonitor import UploadMonitor
        except: pass

        # Make sure we catch and log any exceptions, as they will disappear
        #  silently otherwise (since we're in a background process here)
        try:
            # Store the global exit flag Event object
            self.__dict__['_exit_flag']  = exitflag
            self.__dict__['_first_done'] = firstdone

            # When duplicate is set, generate new randomness per process
            if self.duplicate: self.random = np.random.default_rng(proc_idx)

            # Otherwise randomness should be the same, and we just operate on
            # different sections of the data
            else: self._pidx = proc_idx

            # Start adding batches to the queue
            list(map(queue.put, self._generator))

        except Exception as e:
            self.error(f'Process {proc_idx} exception: {e}')
            raise



    # Cached properties: all return values are cached after first access
    # ==================================================================

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
            A list of started background processes which are operating
            on the Batcher._put method. These background processes should
            not be interacted with directly, as their generated batches
            can be accessed via the Batcher._queue object.

        """
        self.info(f'Starting {self.workers} background processes')

        # Create the spawning context and the queue used to transfer batches
        ctx    = mp.get_context('spawn')
        queue  = self._queue = ctx.Queue(self.max_queue)
        create = lambda *a: ctx.Process(args=a, target=self._put, daemon=True)

        # Create and start the background processes
        jobs = [create(queue, i, self._exit_flag, self._first_done) for i in range(self.workers)]
        [job.start() for job in jobs]

        # Wait a second for them to start, then ensure that they are running
        time.sleep(1)
        if any(not job.is_alive() for job in jobs):
            message = 'Batcher processes are stopping immediately. This is '
            message+= 'likely due to issues pickling the given Dataset. Do '
            message+= 'not run Dataset.generate_samples() on the given object.'
            message+= f'\nExit codes: {[job.exitcode for job in jobs]}'
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


    @cached_property
    def _first_done(self) -> mp.synchronize.Event:# -> threading.Event:
        """Flag to signal the first batch has been generated.

        Notes
        -----
        Forcing all threads to wait for the first batch to be produced
        allows a faster time to first batch, since there is no competition
        for computational resources for the single thread generating it.

        Returns
        -------
        threading.Event
            This uses threading.Event, as we only want to wait for the first
            batch within intra-process threads - not across processes.

        """
        return mp.get_context('spawn').Event()#  threading.Event()


    @cached_property
    def _remainder_lock(self) -> _thread.LockType:
        """ Lock ensuring _remainder is modified by one thread at a time """
        return threading.Lock()


    @cached_property
    def _logger(self):
        """ Create the logging object which writes logs to a file """
        # If a logfile is requested, use a file handler
        if self.logfile is not None:
            # A rotating handler would be nice, but doesn't work in Windows
            handler = logging.FileHandler(self.logfile)

            # If we're in the main process, delete the prior logfile
            if not mp.current_process().daemon:
                Path(self.logfile).write_text('')

        # Otherwise just log to sys.stderr
        else: handler = logging.StreamHandler()

        msg_format = '%(asctime)s | %(process)6d | %(threadName)s'
        msg_format+= ' | %(levelname)7s | %(message)s'
        handler.setFormatter( logging.Formatter(msg_format) )

        logger = logging.getLogger('Batcher')
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.loglevel)
        logger.addHandler(handler)
        return logger
