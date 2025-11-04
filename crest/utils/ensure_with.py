from contextlib import nullcontext
from typing import Callable, ContextManager, Any, Union
import inspect


def ensure_with(
    x : Union[Callable, ContextManager],
) -> ContextManager[Callable[[Any], Any]]:
    """Ensure `x` can be used in a `with` block as a context manager.

    Notes
    -----
    For functions that are given as parameters to other routines, this wrapper
    can allow users to optionally define enter/exit behavior for them. See the
    examples for more detail.

    Parameters
    ----------
    x : Union[Callable, ContextManager]
        The given value can be a regular callable function (e.g. lambda x: x),
        a context manager (whose __enter__ should return a callable), or a
        zero-arg factory that returns such a context manager. Note that this
        function attempts to call `x` with zero arguments if it isn't already
        a context manager object, in order to determine if it's a factory.

    Returns
    -------
    ContextManager[Callable[[Any], Any]]
        Guarantees the returned object can be used as a context manager.
    
    Examples
    --------
    >>> identity = lambda x: x
    >>> with ensure_with(identity) as wrapped:
    ...     print(wrapped is identity)
    True
    >>> @contextmanager
    ... def accumulate_to_disk():
    ...     try:
    ...         accumulated = []
    ...         def collect(i):
    ...             accumulated.append(i)
    ...         yield collect
    ...     finally:
    ...         # Write the accumulated values to disk at exit
    ...         with open('accumulated.txt', 'w+') as f:
    ...             f.write(','.join(map(str, accumulated)))
    ...
    >>> def iterate(function, N):
    ...     with ensure_with(function) as f:
    ...         for i in range(N):
    ...             f(i)
    ...
    >>> iterate(print, 3) # Calls print with no arguments first
    
    0
    1
    2
    >>> iterate(accumulate_to_disk(), 3)
    >>> with open('accumulated.txt', 'r') as f:
    ...     print(f.read())
    0,1,2
    >>> iterate(accumulate_to_disk, 4)
    >>> with open('accumulated.txt', 'r') as f:
    ...     print(f.read())
    0,1,2,3
    
    """

    def requires_no_args(func: Callable) -> bool:
        """ Checks if a function can be called without any arguments """
        try:
            signature = inspect.signature(func)
            for param in signature.parameters.values():
                if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.KEYWORD_ONLY):
                    if param.default is inspect.Parameter.empty:
                        return False  # Found a required argument
            return True               # No required arguments found
        except (ValueError, TypeError): 
            return False              # Failed inspection (e.g. C extension)

    # Checks if given object has enter/exit methods
    is_manager = lambda f: hasattr(f, '__enter__') and hasattr(f, '__exit__')
    
    # If `x` isn't already a context manager
    if not is_manager(x):
        if not callable(x):
            raise TypeError(f'{x=} must be a callable or context manager')

        # Wrap `x` if it requires arguments or it doesn't return a manager
        x = y if requires_no_args(x) and is_manager(y:=x()) else nullcontext(x)
    return x