from collections import defaultdict as dd
from functools import wraps
from inspect import getframeinfo, stack
from pathlib import Path
from typing import Callable
from time import time


def limit_calls(
    limit    : int = 1, 
    timespan : float | None = 1, 
    by_line  : bool = True,
    logger   : bool = True,          
) -> Callable:
    """ Decorator which limits the number of times a function can be called.

    Notes
    -----
    The number of times a function can be called may be conditioned on the 
    elapsed time between calls; i.e. a function can only be called 10 times 
    in the span of 30 seconds. It may also be conditioned on the actual line
    the function is called on, rather than only on the function overall.
    
    This decorator is generally useful for debugging functions which print or
    log some information. By limiting the number of times the function can be
    executed within some span of time, it can make debugging easier by reducing
    noise coming from high-frequency loggers.

    Parameters
    ----------
    limit : int
        The number of times a function can be called. If `timespan=None` is
        given to the decorator, `limit` defines the total number of times the
        function will ever be executed over the life of the program.
    timespan : float | None
        The length of time (in seconds) that the call limit will be conditioned
        on. Note that the length of time is calculated on a rolling basis, 
        meaning the function can only be called the specified number of times 
        within any window of time that is `timespan` seconds long; e.g. if 
        `limit=2; timespan=10` and a function executes at t=[1,4,7,10,12,13]:
        
        - calls at t=1 and t=4 will be executed
        - calls at t=7 and t=10 will not be executed (<= 10 seconds from t=1)
        - call at t=12 will be executed (only t=4 in rolling 10 second window)
        - call at t=13 will not be executed (t=4 and t=12 in the time window)

        By default `timespan=1`, which means that the decorated function is
        only executed (at most) `limit` times within 1 second. `timespan=None`
        indicates the limit is not conditioned on time, and so the limit is
        the total number of calls allowed over the life of the program.
    by_line : bool
        Determines whether the number of calls is tracked by execution line,
        or only on the overall function itself. By default `by_line=True`, and
        so the decorated function will only be allowed to execute `limit` times
        on a specific line - whereas with `by_line=False`, the function will
        only be called `limit` times anywhere in the life of the program.
    logger : bool
        Indicates whether the decorated function is a logger, where the first
        argument may always be interpreted as a string (e.g. `print`). If True
        (default), the following prefix will be added to function calls if any
        calls were skipped: "[<function name><@file:line> skipped <N> times] "
        where <function name> is the name of the decorated function, <N> is the
        number of skipped calls since the last successful execution, and only
        if `by_line=True` will the filename and line number be included. 
        Regardless of the value of `logger`, the number of times the function
        was skipped will also be added as an attribute to the function itself
        as '_call_skips' (if possible).

    Returns
    -------
    Callable
        The decorator to be applied using the requested configuration.
        
    Examples
    --------
    >>> from time import sleep
    >>> @limit_calls(limit=2, timespan=1, logger=False)
    ... def f(): 
    ...     return True
    ...
    >>> for i in range(1, 24):
    ...     if i in [1,4,7,10,12,13,14,15,17,18,20,23]:
    ...         if f():
    ...             print(i, f'called ({f._call_skips} skips)')
    ...         else:
    ...             print(f'skipped {i}')
    ...     sleep(0.1)
    ...
    1 called (0 skips)
    4 called (0 skips)
    skipped 7
    skipped 10
    12 called (2 skips)
    skipped 13
    14 called (1 skips)
    skipped 15
    skipped 17
    skipped 18
    skipped 20
    23 called (4 skips)

    """

    if (timespan is not None) and (timespan <= 0):
        raise ValueError(f'{timespan=} must be >= 0')
        
    def decorator(function: Callable) -> Callable:
        fline = ''
        fname = getattr(function, '__name__', str(function))
        calls = dd(list)
        skips = dd(list)

        @wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal fline
            called_at = time()
            
            # Get the current line number if requested
            if by_line:
                frame = getframeinfo(stack()[1][0])
                fline = f'@{Path(frame.filename).name}:{frame.lineno}'

            # Short-circuit if timespan is None
            if (timespan is None) and (len(calls[fline]) >= limit):
                return

            n_skip = len(skips[fline])
            prefix = f'[{fname}{fline} skipped {n_skip} times]'

            # Remove any expired times
            while len(calls[fline]) and (called_at-calls[fline][0]) > timespan:
                calls[fline].pop(0)

            # If number of executions is less than the limit, call the function
            if len(calls[fline]) < limit:

                # Add the logging prefix if requested and number of skips > 0
                if logger and n_skip:

                    # Find first instance of a string
                    args = list(args)
                    for i, arg in enumerate(args):
                        if isinstance(arg, str):
                            args[i] = f'{prefix} {arg}'
                            break

                    # If no string arguments were found, assume the first arg
                    else:
                        if len(args):
                            args[0] = f'{prefix} {args[0]}'

                        # If no arguments were given, pass a new one
                        else:
                            args = [f'{prefix} ']

                # Add current call and remove skips that have been logged
                calls[fline].append(called_at)
                skips[fline] = []

                # Attempt to add the skip count as a function attribute
                try:    setattr(wrapper, '_call_skips', n_skip)
                except: pass
                return function(*args, **kwargs)

            # Otherwise, add the current call time to the skip list
            skips[fline].append(called_at)

        return wrapper
    return decorator