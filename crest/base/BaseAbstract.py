from collections.abc import Callable
from functools import partial
from itertools import starmap
from typing import get_args, get_origin, _type_repr
from typing import Union, Iterator, TypeVar
from types import UnionType
from abc import ABCMeta, ABC

import linecache
import traceback
import weakref
import inspect
import code
import os


def type_repr(val=None, T=None, _maxdepth=4):
    """ Recursive type representation """
    if isinstance(T, str):        return T
    if T is not None:             return _type_repr(T)
    if isinstance(val, Iterator): return type(val)
    container = _type_repr(type(val)).split('.')[-1]

    # Recurse on val if it's iterable (and not a str)
    if hasattr(val, '__iter__') and not isinstance(val, str) and _maxdepth:
        recurse = partial(type_repr, T=T, _maxdepth=_maxdepth - 1)
        try:
            item = next(iter(getattr(val, 'items', lambda: val)()))
            if hasattr(val, 'items'):
                ele_type = ', '.join(map(recurse, item))
            else:
                ele_type = recurse(item)
        except:
            ele_type = '?'
        return f'{container}[{ele_type}]'
    return container


def equal_tuples(val, T):
    """ Check that val and T are tuples of the same length """
    is_tuple = isinstance(val, tuple) and isinstance(T, tuple)
    return is_tuple and (len(val) == len(T))


def handle_generic(val, T):
    """ Swap generic TypeVar for val type """
    if isinstance(T, TypeVar) and len(T.__constraints__):
        raise NotImplementedError('TypeVar constraints not implemented')
    if isinstance(T, TypeVar):
        T = T.__bound__ or type(val)
    elif equal_tuples(val, T):
        T = tuple(map(handle_generic, val, T))
    elif T is None:
        T = type(None)
    return T


def istype(val, T):
    """ Recursively determine if value matches generic type T """
    # Cannot look inside iterators to verify types, as it would exhaust values
    if isinstance(val, Iterator):
        origin = get_origin(T) or T
        types = get_args(T)
        if origin in [Union, UnionType]:
            return any(istype(val, T) for T in types)
        return isinstance(val, origin)

    T = handle_generic(val, T)

    # Handle a tuple of types
    if equal_tuples(val, T):
        return all(map(istype, val, T))
    elif isinstance(T, tuple):
        return False

    # Try a simple type check, which fails if T is a parameterized generic
    try:
        return isinstance(val, T)
    except TypeError:
        pass

    # Get origin type and parameterized types 
    origin = get_origin(T)
    types = get_args(T)

    # Origin is just a union of types, so we can check for any valid
    if origin in [Union, UnionType]:
        return any(istype(val, T) for T in types)

    # String with class name might be used in the class definition
    if (origin is None) and isinstance(T, str):
        cls_eq = lambda cls: getattr(cls, '__name__', '') == T
        return any(map(cls_eq, (type(val),) + val.__class__.__bases__))

    # Ensure it matches the origin type
    if isinstance(val, origin):

        # If val can't be iterated, i.e. is not a container type like list
        if not hasattr(val, '__iter__'): return True

        # Recursively type check the container elements, replacing
        # T with the type of the first element when necessary
        elems = getattr(val, 'items', lambda: [[v] for v in val])()
        first = (list(elems) + [[]])[0]
        types = list(map(handle_generic, first, types))
        return all(all(map(istype, v, types)) for v in elems)


class EnsureTypes:
    """Ensure type annotations are followed, raising TypeError if not.

    Wrapping with a class rather than a function allows access to the 
    underlying object attributes when a callable object is wrapped.

    """

    def __init__(self, cls_obj: 'BaseAbstract', callable_obj: Callable):
        self._cls_repr = repr(cls_obj)
        self._callable = callable_obj

    def __repr__(self):
        return f'{self._cls_repr}.{self._callable.__code__.co_name}'

    def __call__(self, *args, **kwargs):
        """ Wrap the function with an explicit type checker """

        # Extract the function parameters and respective annotations
        function = self._callable
        annotate = function.__annotations__
        keyvalue = inspect.getcallargs(function, *args, **kwargs)
        keyvalue |= {'return': function(*args, **kwargs)}

        # Iterate over all parameters and verify types match the annotations
        [self.verify_type(value, annotate[key], f'{self} parameter "{key}"')
         for key, value in keyvalue.items() if key in annotate]
        return keyvalue['return']

    def __getattr__(self, attr):
        """ Pass through attribute lookups to the underlying callable """
        return self if attr == '__call__' else getattr(self._callable, attr)

    @classmethod
    def wrap(cls, obj, obj_attr):
        """ Wrap the object attribute if valid, and return it otherwise """
        # 1) not EnsureTypes; 2) callable; 3) not bytecode; 4) annotated
        if (not isinstance(obj_attr, cls)
                and callable(obj_attr)
                and hasattr(obj_attr, '__code__')
                and getattr(obj_attr, '__annotations__', {})):
            return cls(obj, obj_attr)
        return obj_attr

    @classmethod
    def verify_type(cls, obj, annotation, label):
        """ Raise TypeError if obj type does not match the given annotation """
        try:
            invalid_type = not istype(obj, annotation)
        except TypeError:
            raise TypeError(f'{label} annotation ' +
                            f'"{annotation}" is not a valid type annotation')

        if invalid_type:
            req = type_repr(T=annotation)
            typ = type_repr(obj)
            msg = f'{label} must be of type {req}, but found type {typ}'
            raise TypeError(msg)


class BaseAbstract(ABC):
    """ Base class for any other 'Base' classes. """

    def __repr__(self):
        return self.__class__.__name__

    def __getattribute__(self, name):
        """ Provides type checking for class functions that use annotations """
        attr = object.__getattribute__(self, name)
        return attr if name.startswith('__') else EnsureTypes.wrap(self, attr)

    def __new__(cls, *args, **kwargs):
        """ Called whenever a new inheriting class object is instantiated """
        # Store a weakref of the object to allow tracking object persistance
        obj = super().__new__(cls)
        cls._refs[id(obj)] = obj
        return obj

    def __init_subclass__(cls, *args, **kwargs):
        """ Called when an inheriting class is defined.
            Wraps __init__ with type checking, and allows 
            __post_init__ functions in inheriting classes.
        """

        def init_decorator(init):
            def __init__(self, *args, **kwargs):
                init(self, *args, **kwargs)
                # Only call for the final __init__ in the inheritance stack
                if type(self) is cls: self.__post_init__()

            return __init__

        type_checked = EnsureTypes.wrap(cls, cls.__init__)
        cls.__init__ = init_decorator(type_checked)
        cls._refs = weakref.WeakValueDictionary()

    def __post_init__(self):
        """ Allows inheriting classes to define a function that runs after
            the __init__ method; mainly useful for Base classes to force
            children to perform some operations after initialization """
        pass

    @classmethod
    def load(cls, obj, *args, **kwargs):
        """ Wrap an object with the parent class if it isn't already one """
        return obj if isinstance(obj, cls) else cls(obj, *args, **kwargs)

    @classmethod
    def interactive(cls, env={}, style='monokai'):
        """ Start an interactive console wherever this function is called. 

        Parameters
        ----------
        env   : dict
            Any extra objects to include in the terminal environment.
        style : str
            Pygments style name for code highlighting.

        """

        def get_context(frame, n_lines=20):
            """ Get code context at given frame """
            name = frame.f_code.co_filename 
            line = frame.f_lineno + 1 
            size = len(str(line-1))

            getline = lambda i: f'{i:<{size}} {linecache.getline(name, i)}'
            context = f'{name}:{line-1} called interactive:'
            divider = ''.join(['-']*len(context))
            srccode = ''.join(map(getline, range(max(0, line-n_lines), line)))
            return '\n'.join(['', divider, context, '', srccode, divider, ''])

        # Include variables from the frame this function was called in
        callframe = inspect.currentframe().f_back
        variables = callframe.f_globals | callframe.f_locals | env

        # Define flag to indicate exception in the console
        global forcestop
        forcestop = False

        # Show code context as the console banner
        try:
            banner = get_context(callframe)

            # Add syntax highlighting
            try:
                from pygments import highlight
                from pygments.lexers import Python3Lexer
                from pygments.styles import get_style_by_name
                from pygments.formatters import Terminal256Formatter
                srcfmt = Terminal256Formatter(style=get_style_by_name(style))
                banner = highlight(banner, Python3Lexer(), srcfmt)

                # On windows, for some versions of python, ANSI escape codes
                # are not processed by the console. A solution is to run 
                # `os.system('')`, which will indirectly set the windows
                # terminal ENABLE_VIRTUAL_TERMINAL_PROCESSING flag - thus
                # enabling the parsing of control sequences. For more details,
                # see https://stackoverflow.com/a/64222858
                if os.name == 'nt': 
                    os.system('')

                    # Another potential source of breaking ANSI codes can 
                    # come from using colorama, or importing anything that
                    # uses colorama on initialization (e.g. tqdm). The fix
                    # for this is to call `colorama.deinit()`; however it
                    # has been noted that although calling this fixes ANSI
                    # ANSI code parsing in some cases, it can also break it
                    # in other cases. To that end, the following code is 
                    # left commented out by default; users can choose to 
                    # enable it or use it as reference when fixing an issue
                    # in their specific environment. For more details, see
                    # https://github.com/tqdm/tqdm/issues/678#issuecomment-70662706
                    # import colorama
                    # colorama.deinit()
                    
            except ImportError: pass
        except Exception as e: banner = f'\n{e}\n{traceback.format_exc()}'

        try:
            # Import readline if available to pull interpreter command history
            import readline, rlcompleter
            variables = locals() | variables

            # Allow tab completion
            # https://tiswww.case.edu/php/chet/readline/readline.html
            readline.set_completer(rlcompleter.Completer(variables).complete)
            readline.parse_and_bind('tab: complete')

            # Read previous command history
            readline.read_history_file()

            def follow_history(*args, **kwargs):
                """ implements operate-and-get-next: allows successive commands
                    from history to be selected (i.e. after navigating history
                    and executing a command, the cursor will still be at the 
                    same location in the console history) """

                def _handle_exc(e):
                    # if there's an exception, print info, set flag, and exit
                    print(f'\n{traceback.format_exc()}\nException in ' +
                          f'BaseAbstract.interactive.follow_history: {e}')
                    global forcestop
                    forcestop = True
                    raise EOFError

                def _set_cursor(index, *args, **kwargs):
                    # Set the index and remove this function from the queue
                    try:
                        terminal = readline.rl.mode
                        terminal._history.set_history_cursor(index)
                        terminal.process_keyevent_queue.pop(-1)
                        return terminal.process_keyevent(*args, **kwargs)
                    except Exception as e:
                        _handle_exc(e)

                try:
                    # Get the current index in history and total length
                    terminal = readline.rl.mode  # e.g. pyreadline3.modes.emacs
                    hist_len = len(terminal._history.history)
                    hist_idx = terminal._history.get_history_cursor() + 1

                    # If we're examining somewhere in previous command history
                    if hist_idx < hist_len:
                        prev_cmd = terminal._history.get_history_item(hist_idx)
                        curr_cmd = terminal.l_buffer.get_line_text()

                        # If command is unchanged, keep history cursor at index
                        if curr_cmd == prev_cmd:
                            set_cursor = partial(_set_cursor, hist_idx - 1)
                            terminal.process_keyevent_queue.append(set_cursor)
                except Exception as e:
                    _handle_exc(e)

                # Otherwise just accept the line as usual
                return terminal.accept_line(*args, **kwargs)

            # Bind the enter key so that we can execute multiple commands
            # from history without needing to press up for each command 
            readline.rl.mode._bind_key('return', follow_history)

            class InteractiveConsole(code.InteractiveConsole):
                """ Fix InteractiveConsole raw_input() ANSI coloring.

                    pyreadline breaks full RGB ANSI colors for input().
                    We circumvent pyreadline's implementation by printing
                    the input prompt separately from the actual input call. 
                    Note that this fixes the console prompt, but not input().
                """
                def raw_input(self, prompt=''):
                    print(prompt, end='')
                    return input()

        except ImportError: from code import InteractiveConsole

        # Set the console prompt colors
        import sys
        sys.ps1 = '\x1b[38;5;197m>>>\x1b[0m '
        sys.ps2 = '\x1b[38;5;141m...\x1b[0m '

        # Start the console
        banner += '\nCTRL-Z resumes execution, quit() halts execution'
        try: InteractiveConsole(variables).interact(banner=banner)

        # Ensure we write the command history after exiting
        finally:
            try:
                readline.write_history_file()
            except Exception as e:
                print(f'Failed to write history: {e}')

        # Raise a SystemExit if the exception flag was set while running
        if forcestop: raise SystemExit

    @classmethod
    def verify_type(cls, obj, annotation, label=''):
        """ Allow a class to verify arbitrarily complex types manually """
        EnsureTypes.verify_type(obj, annotation, label or f'{cls}: {obj}')
