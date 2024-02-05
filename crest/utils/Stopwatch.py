from collections.abc import Callable
from itertools import starmap
from numbers import Number
from psutil import Process  
from math import floor, log10
from time import sleep, time
import gc 


def get_memory(priority: list[str] = ['uss', 'vms', 'rss']) -> int:
    """Get memory usage of current process. 
    
    Parameters
    ----------
    priority : list[str]
        The first key in this list that is successfully retrieved is
        the type of memory value returned.

    Returns
    -------
    int
        Current memory usage.

    .. _priority options:
        https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info

    """
    process = Process()
    try:    usage = process.memory_full_info()
    except: usage = process.memory_info()

    for key in priority:
        if hasattr(usage, key):
            return getattr(usage, key)
    raise Exception(f'No valid keys available in {usage}')



class Stopwatch:
    """Context manager to time code blocks.
    
    Parameters
    ----------
    prefix  : str
        String to prefix the logged output with; i.e. f'{prefix}: {metrics}'.
    logger  : Callable
        Function which is passed the output string. By default this is just
        'print', which logs to the console - but any callable could be given,
        such as a logging.info function. 
    timer   : Callable
        Function called upon entering the Stopwatch context manager, and when
        exiting it - where the difference is logged as the interval of time 
        that passed. By default this is time.time, but other options may give 
        more sensible results in some cases (e.g. time.perf_counter).
    memory  : Callable
        Function called upon entering the Stopwatch context manager, and when
        exiting it - where the difference is logged as the change in memory 
        (dMem). By default this uses psutil to retrieve (in order of priority):
        uss, vms, or rss; but any other callable that returns memory usage 
        could be given instead. 
    metrics : dict[str, Callable]
        A dictionary of any other functions that should be called when entering
        and exiting the context manager, and then the difference logged. The 
        keys in this dictionary will be used to label the logged difference.
        For example, the metric for change in memory could be defined as::

            {'dMem': (lambda: psutil.Process().memory_full_info().uss)} 
    
    formats : dict[str, dict]
        A dictionary of keys matching those in `metrics`, and values being the
        keyword arguments for formatting the respective metrics with. Valid 
        keyword arguments are 'units' and 'divisor'; see Stopwatch.readable for
        the docstring.
    samples : int
        Number of times to repeatedly call metric functions, in order to obtain
        an averaged value over time. Should be used in combination with the 
        `delay` parameter in order to add a time delay between function calls.
    delay   : Number
        Amount of time (in seconds) to wait between function calls when taking
        multiple samples. If `samples` <= 1, this parameter has no effect.  
    silent  : bool
        If true, calculate the change in metric values but do not log outputs.
        Note that these difference values are stored in `Stopwatch.deltas`, 
        which can be accessed after the Stopwatch context manager has exited.
    stop_gc : bool
        Stops garbage collection while inside the Stopwatch context manager,
        which can sometimes be useful for getting a more accurate estimate
        of elapsed time and memory usage.

    Examples
    --------
    >>> import time
    >>> with Stopwatch():
    ...   time.sleep(1)
    time=1 seconds  dMem=0 B
    >>> with Stopwatch('using time.perf_counter', timer=time.perf_counter):
    ...   time.sleep(1)
    using time.perf_counter: time=1.01 seconds  dMem=0 B

    """

    GC_DISABLE = 0

    default_fmt = {
        'time': {'units': 'time'},
        'dMem': {'units': 'byte'},
    }

    def __init__(self, 
        prefix  : str = '', 
        logger  : Callable = print, 
        timer   : Callable = time,
        memory  : Callable = get_memory,
        metrics : dict[str, Callable] = {},
        formats : dict[str, dict]     = {}, 
        samples : int = 1,
        delay   : Number = 0,
        silent  : bool = False,
        stop_gc : bool = False,
    ):
        self.prefix  = prefix
        self.logger  = logger 
        self.metrics = {'dMem': memory} | metrics
        self.formats = self.default_fmt | formats
        self.samples = samples
        self.delay   = delay 
        self.silent  = silent
        self.stop_gc = stop_gc

        # Time is handled separately to avoid influence by other metrics
        self.timer = timer


    def __enter__(self):
        """ Begin tracking the requested metrics """
        # Only disable (and later re-enable) the gc if it's currently enabled
        self.stop_gc &= gc.isenabled()
        if self.stop_gc: gc.disable()

        # Store the initial values for all metrics, fetching time last
        self.start = {k: self._sample(v) for k,v in self.metrics.items()}
        self.start|= {'time': self.timer()}
        return self 


    def __exit__(self, *args, **kwargs):
        """ Finish tracking metrics, calculate deltas, and log if requested """
        self.finish = {'time': self.timer()}
        self.finish|= {k: self._sample(v) for k,v in self.metrics.items()}
        self.deltas = {k: self.finish[k] - self.start[k] for k in self.finish}
        if self.stop_gc: gc.enable()

        # Format the metric delta into the final output string
        fmt = lambda k,v: f'{k}={self.readable(v, **self.formats.get(k, {}))}'

        if not self.silent: 
            message = ': '.join(
                ([str(self.prefix)] if self.prefix else []) +
                ['  '.join(starmap(fmt, self.deltas.items()))]
            )
            try:    self.logger(message, stacklevel=2)
            except: self.logger(message)


    def __getitem__(self, key):
        """ Return the delta value for the requested metric """
        if hasattr(self, 'deltas'):
            if key in self.deltas:
                return self.deltas[key]
            raise Exception(f'Unknown key "{key}": {list(self.deltas)}')
        raise Exception('Stopwatch has not yet exited or calculated metrics')


    @staticmethod
    def readable( 
        value   : Number, 
        units   : str | list[str] = 'size', 
        divisor : Number | None   = None,
    ) -> str:
        """Format a value to have reasonable rounding and units.
        
        Notes
        -----
        Reduces resolution by one decimal place every multiple of 10; e.g.:

        - [0.1,   1) -> 3 decimal places
        - [  1,  10) -> 2 decimal places
        - [ 10, 100) -> 1 decimal place
        
        100 onwards it will use 0 decimal places.  

        Also attaches units such that the available units list is iterated 
        through until the number can no longer be completely divided by the 
        divisor. For example, with units='time' and divisor=60, the number 
        (given initially as seconds) are converted to minutes for `value` 
        greater than 100 seconds, and then from minutes to hours after 100 
        minutes.

        Parameters
        ----------
        value   : Number
            The number to get the string representation for.
        units   : str | list[str]
            Units to label the number with; one of ['time', 'byte', 'size'].
            A custom list of unit strings can also be provided.
        divisor : Number | None
            Signifies moving to the next unit when the current number
            can be completely divided by the divisor. If divisor is None
            (default) then a value is chosen based on the given units.

        Returns
        -------
        str
            String which rounds the given value to a reasonable
            number of decimal places and attaches the correct units.
        
        Examples
        --------
        >>> Stopwatch.readable(12345)
        '12.3K'
        >>> Stopwatch.readable(12345, 'byte')
        '12.1 KB'
        >>> Stopwatch.readable(12345, 'time')
        '206 minutes'
        >>> Stopwatch.readable(1234.5, 'time')
        '20.6 minutes'
        >>> Stopwatch.readable(123.45, 'time')
        '123 seconds'
        >>> Stopwatch.readable(123456, 'time')
        '34.3 hours'

        """

        divisor = divisor or {
            'time' : 60,
            'byte' : 1024,
            'size' : 1000,
        }.get(units if isinstance(units, str) else 'size', 1000)

        units = units if isinstance(units, list) else {
            'time' : [' seconds', ' minutes', ' hours'],
            'byte' : [' B', ' KB', ' MB', ' GB', ' TB'],
            'size' : ['', 'K', 'M', 'G', 'T'],
        }.get(units, [])

        def fmt(value, units):
            """ Recurse until value is small enough or we run out of units """
            add_unit = lambda v: ''.join([v]+units[:1])
            if value == 0: return add_unit('0')
            decimals = max(-1, -floor(log10(abs(value))) + 2)

            if (decimals == -1) and (len(units) > 1):
                return fmt(value/divisor, units[1:])

            value = f'{value:.{max(0, decimals)}f}'
            if '.' not in value: value += '.'
            return add_unit(value.rstrip('0').rstrip('.'))
        return fmt(value, units)


    def _sample(self, function: Callable) -> Number:
        """ Average multiple function values with delays in between calls """
        if self.samples > 1:
            call = lambda: sleep(self.delay) or function()
            return sum(call() for _ in range(self.samples)) / self.samples
        return function()