from collections.abc import Callable
from numbers import Number  
from time import perf_counter, time as timetime
from math import floor, log10


class Stopwatch:
    """Context manager to time code blocks.
    
    Parameters
    ----------
    label : str
        String to prefix the logged output with; i.e. f'{label}: {interval}'.
    log   : Callable
        Function which is passed the output string. By default this is just
        'print', which logs to the console - but any callable could be given,
        such as a logging.info function. 
    time  : Callable
        Function which is called at the start of the timer, and at the end
        of the timer - where the difference is used as the interval of time
        that passed. By default this is time.perf_counter, but other options
        may give more sensible results in some cases (e.g. time.time).

    Examples
    --------
    >>> import time
    >>> with Stopwatch():
    ...   time.sleep(1)
    1.01 seconds
    >>> with Stopwatch('usimg time.time', time=time.time):
    ...   time.sleep(1)
    1.00 seconds

    """
    def __init__(self, label='', log=print, time=timetime):
        self.label = label
        self.log   = log 
        self.time  = time 


    def __enter__(self):
        self.start = self.time()
        return self 


    def __exit__(self, *args, **kwargs):
        runtime = self.runtime = self.time() - self.start
        outputs = [self.label] if self.label else []
        outputs+= [self._round(runtime)]
        self.log(': '.join(map(str, outputs)))


    def _round(self, i: Number, _units=['seconds', 'minutes', 'hours']) -> str:
        """Give reasonable rounding and units for a time interval.
        
        Notes
        -----
        Reduces resolution by one decimal place every multiple of 10; e.g.:
            - [0.1,   1) -> 3 decimal places
            - [  1,  10) -> 2 decimal places
            - [ 10, 100) -> 1 decimal place
        100 onwards it will use 0 decimal places.  

        Also attaches units such that seconds are converted to minutes 
        after 100 seconds, and minutes to hours after 100 minutes. 

        Parameters
        ----------
        i : Number
            Time interval to get the string representation for.

        Returns
        -------
        str
            String which rounds the given value to a reasonable
            number of decimal places and attaches the correct units.
        
        """
        if i <= 0: return '0 seconds'
        decimals = max(0, -floor(log10(i)) + 2)

        if (decimals == 0) and (len(_units) > 1):
            return self._round(i/60., _units[1:])
        return f'{i:.{decimals}f} {_units[0]}'
