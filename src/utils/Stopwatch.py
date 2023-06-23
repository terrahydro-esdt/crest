from collections.abc import Callable
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

    .. _Key options:
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
        keys in this dictionary will be used to label when logging.
    formats : dict[str, dict]
        A dictionary of keys matching those in `metrics`, and values being the
        keyword arguments for formatting the respective metrics with. Valid 
        keyword arguments are 'units' and 'divisor'; see Stopwatch.readable for
        docstring.
    samples : int
        Number of times to repeatedly call metric functions, in order to obtain
        an averaged value over time. Should be used in combination with the 
        `delay` parameter in order to add a time delay between function calls.
    delay   : Number
        Amount of time (in seconds) to wait between function calls when taking
        multiple samples. If `samples` <= 1, this parameter has no effect.  


    Examples
    --------
    >>> import time
    >>> with Stopwatch():
    ...   time.sleep(1)
    1.00 seconds
    >>> with Stopwatch('using time.perf_counter', timer=time.perf_counter):
    ...   time.sleep(1)
    1.01 seconds

    """
    GC_DISABLE = 0

    def __init__(self, 
        prefix  : str = '', 
        logger  : Callable = print, 
        timer   : Callable = time,
        memory  : Callable = get_memory,
        metrics : dict[str, Callable] = {},
        formats : dict[str, dict]     = {}, 
        samples : int = 1,
        delay   : Number = 0,
    ):
        self.prefix  = prefix
        self.logger  = logger 
        self.metrics = metrics | {'dMem': get_memory}
        self.formats = formats | {'time' : {'units': 'time', 'divisor': 60}}
        self.samples = samples
        self.delay   = delay 

        # Time is handled separately to avoid influence by other metrics
        self.timer = timer


    def __enter__(self):
        self.start = {k: self.sample(v) for k,v in self.metrics.items()}
        self.start|= {'time': self.timer()}
        Stopwatch.GC_DISABLE += 1
        gc.disable()
        return self 


    def __exit__(self, *args, **kwargs):
        self.finish = {'time': self.timer()}
        self.finish|= {k: self.sample(v) for k,v in self.metrics.items()}

        fmt_out = lambda k, v: self.readable(v, **self.formats.get(k, {}))
        deltas  = {k: self.finish[k] - self.start[k] for k in self.finish}
        outputs = [self.prefix] if self.prefix else []
        outputs+= ['  '.join([f'{k}={fmt_out(k, v)}' for k,v in deltas.items()])]
        self.logger(': '.join(map(str, outputs)))

        Stopwatch.GC_DISABLE -= 1
        if Stopwatch.GC_DISABLE == 0:
            gc.enable()
        assert(Stopwatch.GC_DISABLE >= 0)


    @staticmethod
    def readable( 
        value   : Number, 
        divisor : Number = 1000,
        units   : str    = 'byte', 
    ) -> str:
        """Return a string with reasonable rounding and units.
        
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
        divisor : Number
            Signifies moving to the next unit when the current number
            can be completely divided by the divisor. 
        units   : str
            Units to label the number with; one of ['time', 'byte', 'size'].

        Returns
        -------
        str
            String which rounds the given value to a reasonable
            number of decimal places and attaches the correct units.
        
        """
        units = {
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


    def sample(self, function: Callable) -> Number:
        """ Average multiple function values with delays in between calls """
        if self.samples > 1:
            call = lambda: sleep(self.delay) or function()
            return sum(call() for _ in range(self.samples)) / self.samples
        return function()