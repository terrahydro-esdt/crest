from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections.abc import Callable
from threading import Event
from numbers import Number
from logging import Logger, getLogger
import traceback, time

from crest.utils import Stopwatch


class ThreadedFunction(set):
    """ Encapsulates mapping a function over a set of items using multiple
        threads, while providing some additional logistical support over 
        the native Executor.map function.

        Tasks are returned as a set when waiting for them to complete, and 
        so this class wraps the python set type in order to easily track 
        and update the tasks as they complete.

        Parameters
        ----------
        function : Callable
            The function to apply to any items given to this class.
        threads  : int
            How many threads should be used to execute tasks.
        capacity : int 
            Number of tasks that can be submitted for execution, with 
            capacity <= 0 allowing an unlimited number of tasks to be
            submitted. See __call__ for details on how capacity 
            affects operation.
        timeout  : Number | None
            How long to wait for tasks to be completed when checking for 
            newly completed tasks. If None is given, __iter__ will block
            until at least one task completes. 
        exitflag : Event | None
            Event object used to signal that all threads should stop working.
        logger   : Logger | None
            Logger object to generate logs.

    """

    def __init__(self, 
        function : Callable, 
        threads  : int = 3,
        capacity : int = 6,
        timeout  : float  | None = 0.01, 
        exitflag : Event  | None | Callable = None,
        exitset  : Event  | None | Callable = None,
        logger   : Logger | None = None,
    ):
        super().__init__()
        self.function = function
        self.threads  = threads
        self.capacity = capacity if threads >= 1 else 1
        self.timeout  = timeout 
        self.logger   = logger or getLogger(function.__name__)

        _event = Event()
        self.exitflag: Callable = getattr(exitflag or _event, 'is_set', exitflag)
        self.exitset:  Callable = getattr(exitset or exitflag or _event, 'set', exitset or _event.set)

        # Count of total tasks submitted, and thread pool
        self._items = []
        self._total = 0
        self._pool  = ThreadPoolExecutor(**{
            'max_workers'        : threads, 
            'thread_name_prefix' : function.__name__,
        }) if threads > 0 else None
    

    def __next__(self): 
        if self.threads > 0: raise NotImplementedError
        if len(self):
            args, kwargs = self._items.pop()
            self.pop()
            return self._execute(*args, **kwargs)


    def __iter__(self):
        """ Retrieve completed tasks if there are any available, remove them
            from the task set, and return in the order they were submitted. """
        if self.threads < 1:
            for i in range(len(self._items)):
                yield next(self)
        else:
            try: 
                done, _ = wait(self, self.timeout, FIRST_COMPLETED)
                self.difference_update(done)
                yield from sorted(done, key=lambda d: d.order)
            except TimeoutError: pass


    def __call__(self, *args, **kwargs):
        """Submit args/kwargs to be executed as a task.

        Notes
        -----
        When the task set is full (i.e. there are currently `capacity` tasks
        waiting to be completed) and a new task is given for execution, this
        function will block until one of the existing tasks completes AND is 
        removed from the task set via __iter__. Note that this means if an 
        task is submitted in the main thread and the set is currently full,
        the program will deadlock since completed tasks will never be removed
        from the task set via __iter__. This can be avoid by either:
        - initializing ThreadedFunction with capacity <= 0 (unlimited tasks)
        - checking that the task set has room to submit new tasks before 
          calling on the given task. The number of tasks currently in the set
          can be checked by calling len on the ThreadedFunction object.

        """   
        continuous = kwargs.pop('continuous', False)
        if self._pool is None:
            self._items.append( [args, kwargs] )
            self.add(self._total)
            self._total += 1
            return

        try: 
            # Wait for capacity in the task set or exit to be signaled
            loops = 1
            while (not self.exitflag()) and self.is_full():
                if continuous: return 

                time.sleep(1) 
                if loops % 10 == 0:
                    self.logger.debug(f'Waiting to add more {self.name} tasks')
                loops += 1 

            # If exit not signaled, submit the job and track its order
            if not self.exitflag(): 
                future = self._pool.submit(self._execute, *args, **kwargs)
                future.order = self._total
                self._total += 1
                self.add(future)
            else: 
                self.close()

        except Exception as e:
            if 'new futures after shutdown' not in str(e):
                msg = f'Exception in __call__: {e}\n{traceback.format_exc()}'
                self.logger.error(msg)
                self.close()


    @property
    def name(self) -> str:
        """ Name of the function being threaded """
        return self.function.__name__


    def is_full(self) -> bool:
        """ Checks if this container is at capacity """
        return 0 < self.capacity <= len(self)


    def close(self):
        """ Attempt to gracefully clean up the background threads """
        # Need to wrap _everything_ in try/except since we may be in __del__
        def _exit():
            try:    self.exitset()
            except Exception as e: self.logger.debug(f'Exception in exitset: {e}')
            try:    self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception as e: self.logger.debug(f'Exception in pool.shutdown: {e}')
        try:
            with Stopwatch(f'Closed {self.name} pool', self.logger.debug):
                _exit()
        except: _exit()


    def _execute(self, *args, **kwargs):
        """ Log any errors which result when calling the function """
        try: 
            if self.exitflag(): return
            with Stopwatch(f'\tCompleted {self.name}', self.logger.debug):
                return self.function(*args, **kwargs)

        except Exception as e:
            if 'cannot schedule new futures' not in str(e):
                msg = f'Exception in {self.name}: {type(e)}\n{e}\n{traceback.format_exc()}'
                self.logger.error(msg)
                self.close()
                raise

        finally: 
            if self.exitflag():
                self.close()
