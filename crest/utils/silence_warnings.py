from collections.abc import Callable
import functools
import warnings


def silence_warnings(function: Callable) -> Callable:
    """Simple decorator to silence warnings coming from a function.
    
    Parameters
    ---------
    function : Callable
        Function to decorate.
    
    Returns
    -------
    Callable
        Decorated function which silences warnings from the original function.
    
    """
    
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') 
            return function(*args, **kwargs)
    return wrapper 
