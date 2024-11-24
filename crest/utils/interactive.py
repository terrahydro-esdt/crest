"""

This snippet contains functions that enable starting an interactive console
anywhere in a python execution path. Users can simply call `interactive()`
where they want the console to start, and all local and global variables
from that context will be available within the console session. 

In addition, the `interactive_exceptions` decorator can be used to decorate 
any function, in order to enable starting an interactive console session if
and when an exception is raised by the decorated function. The console will
start where the exception was raised, thus allowing users to easily debug.

Contents summary:
    - interactive_exceptions (function: Callable)
        Decorator that starts a console session where any exceptions occur 
        within the decorated function, to enable interactive debugging.

    - interactive (environment: dict={}, n_prior_frames: int=0, frame=None)
        Function that can be called anywhere, which will start a console
        that has access to all local and global variables where called.

    - get_call_frame (n_prior_frames: int = 0):
        Function that returns a frame object relative to where it's called.
    
    - get_frame_vars (frame: PyFrameObject)
        Function that returns a frame's local and global variables.

    - get_source_code (frame: PyFrameObject, n_lines: int = 20)
        Function that returns a frame's source code context as a string.
    
    - InteractiveConsole
        Class which patches various issues with `code.InteractiveConsole`,
        and adds some nice features to the console functionality.

See docstrings for more details.

"""

from collections.abc import Callable
from functools import partial, wraps
from linecache import getline 

import traceback
import inspect
import code
import sys
import os
import re



def interactive_exceptions(function: Callable) -> Callable:
    """ Decorator that starts a console if an exception occurs.
    
    Parameters
    ----------
    function : Callable
        The function that should be wrapped by this decorator.

    Returns
    -------
    Callable
        A decorated function which, if an exception occurs within
        the original function, will start an interactive console
        at the location the exception was raised.

    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Pass-through to the function, unless an exception occurs
        try: return function(*args, **kwargs)
        
        # Skip starting the console if the exception is manually triggered
        except KeyboardInterrupt: raise

        # All other exceptions start the interactive console where the 
        # exception occurred, and re-raises the exception after completion
        except Exception as e: 
            message = f'Exception was raised by {function}'
            print(f'\n{message}\n' + '-'*len(message))
            traceback.print_exc()

            # Get the frame where the exception occurred
            try:    frame = e.__traceback__.tb_next.tb_frame
            except: frame = None

            # Add the Exception and the function to the console environment
            interactive(frame=frame, n_prior_frames=1, environment={
                'exception' : e,
                'function'  : function,
            })

            # re-raise the same error on console exit
            raise
    return wrapper



def interactive(environment: dict={}, n_prior_frames: int=0, frame=None):
    """ Start an interactive console wherever this function is called.

    Parameters
    ----------
    environment : dict
        Any extra objects to include in the console environment.
    n_prior_frames : int
        Extra frames to rewind when getting the calling context for this
        function. If `interactive` is called directly, this value should
        be 0 to use the context where it was called (default); if this
        function is e.g. called by a helper function and so the desired
        context is where the helper was used at, the helper should use
        `interactive(n_prior_frames=1)`; etc.
    frame : PyFrameObject
        Rather than fetching the frame context within the current stack, 
        a frame object can be explicitly given as the console context.

    """

    # On windows, for some versions of python, ANSI escape codes are not
    # processed by the console. A solution is to run `os.system('')`, 
    # which will indirectly set the windows terminal flag that enables the
    # parsing of control sequences (ENABLE_VIRTUAL_TERMINAL_PROCESSING). 
    # For more details, see https://stackoverflow.com/a/64222858
    if os.name == 'nt': os.system('')

    # Another potential source of breaking ANSI codes can come from using 
    # colorama, or importing anything that uses colorama on initialization 
    # (e.g. tqdm). The fix for this is to call `colorama.deinit()`; however
    # it has been noted that although calling this fixes ANSI code parsing 
    # in some cases, it can also break it in others. With that in mind, the
    # following code is left commented out by default; users can choose to 
    # enable it themselves, or use it as reference when fixing an issue in
    # their specific environment. See the relevant issue on github for more
    # detail: https://github.com/tqdm/tqdm/issues/678#issuecomment-70662706
    # import colorama; colorama.deinit()

    # Include variables from the frame this function was called
    # in, and show the code context as the console banner
    cframe = frame or get_call_frame(n_prior_frames + 1)
    banner = get_source_code(cframe)
    f_vars = get_frame_vars(cframe, **environment)

    # Set the console prompt colors and add banner information
    sys.ps1 = '\x1b[38;5;197m>>>\x1b[0m '
    sys.ps2 = '\x1b[38;5;141m...\x1b[0m '
    banner += '\nCtrl-z resumes execution, quit() halts execution\n'

    # Add helper functions and the call frame itself to the frame variables
    f_vars.update({
        'get_call_frame'  : get_call_frame,
        'get_source_code' : get_source_code,
        'get_frame_vars'  : get_frame_vars, 
        'console_frame'   : cframe,
    })

    # Initialize the console and begin the interactive session
    InteractiveConsole(f_vars).interact(banner=banner)



def get_call_frame(n_prior_frames: int = 0):
    """ Get a frame object relative to where this function is called.
    
    Parameters
    ----------
    n_prior_frames : int
        Number of frames to rewind. A value of 0 (default) returns the 
        context where this function was directly called at; a value of 1 
        returns the context of the function that called this function; etc.

    Returns
    -------
    PyFrameObject
        Returns the frame object which is `n_prior_frames` prior to the
        context where this function was called.
    
    """

    # Rewind the current frame back one step to get the frame where this
    # function was called; then any additional steps that were requested
    frame = inspect.currentframe()
    for _ in range(n_prior_frames + 1): 
        frame = frame.f_back
    return frame



def get_frame_vars(frame, **variables) -> dict:
    """ Get the local and global variables from a frame's context.
    
    Parameters
    ----------
    frame : PyFrameObject
        Frame object (e.g. from `inspect.currentframe()`) to use as the 
        context from which to retrieve the local and global variables.
    **variables
        Any additional variables to include in the returned dictionary.

    Returns
    -------
    dict
        Dict containing local and global variables in the frame context,
        as well as the frame itself stored in the 'frame' key.
        
    """

    # Return all variables in the frame context, and the frame itself
    variables.update(frame.f_globals)
    variables.update(frame.f_locals)
    variables.update({'frame':frame})
    return variables



def get_source_code(frame, n_lines: int = 20) -> str:
    """ Get the source code context at the given frame. 
    
    Notes
    -----
    If the pygments library is available, the returned code string will
    inlude syntax highlighting.

    Parameters
    ----------
    frame : PyFrameObject
        Frame object (e.g. from `inspect.currentframe()`) to use as the 
        context from which to retrieve the source code.
    n_lines : int
        Number of source code lines from the frame context to return.
    
    Returns
    -------
    str
        String which shows the source code context from the given frame.
        
    """

    try:        
        # Get the filename and line number for the frame
        filename = frame.f_code.co_filename 
        line_num = frame.f_lineno 
        num_size = len(str(line_num))

        # Helper to fetch the file line text, and include a line number
        get_line = lambda i: f'| {i:>{num_size}} | {getline(filename, i)}'

        # Get all lines from the source file to avoid e.g. starting the 
        # code syntax highlighting in the middle of a multi-line comment
        source = list(map(get_line, range(1, line_num+1)))

        # Create the necessary context strings
        divide = '=' * min(120, max(map(len, source[-n_lines:])))
        header = f'{filename:^{len(divide)}}'
        source = ''.join(source).strip()

        # Add syntax highlighting if pygments is available
        try:
            from pygments import highlight
            from pygments.lexers import Python3Lexer
            from pygments.styles import get_style_by_name
            from pygments.formatters import Terminal256Formatter
            style = get_style_by_name('monokai')
            t_fmt = Terminal256Formatter(style=style)
            lexer = Python3Lexer()
            color = partial(highlight, lexer=lexer, formatter=t_fmt)

            divide = color(divide).strip()
            header = color(header).strip()
            source = color(source).strip()
        except ImportError: 
            header = f'Install pygments for syntax highlighting!\n{header}'
    except Exception as e: 
        return f'\n{traceback.format_exc()}\nError in get_source_code: {e}'

    # Include only the requested number of lines
    source = source.split('\n', max(0, line_num - n_lines))[-1]
    pieces = [divide, header, divide.replace('=', '_'), source, divide]
    return '\n'.join([''] + pieces + [''])



class InteractiveConsole(code.InteractiveConsole):
    """ Fixes various issues with parent class, and adds functionality.

    Notes
    -----
    If available, the readline and rlcompleter libraries are used to allow:
    - previous console command history
    - using tab to complete variable names
    - using tab to print object attrs/functions (e.g. obj.<tab>)
    - restoring history to the same place after executing previous command,
      so multiple commands can be executed without needing to scroll back 
      for each. Note: Windows only, due to pyreadline3 dependency

    .. _readline reference: 
        https://tiswww.case.edu/php/chet/readline/readline.html

    """

    def __init__(self, variables: dict, *args, **kwargs):
        """ Performs various operations for a better console experience """
        super().__init__(variables, *args, **kwargs)

        # Define flag to indicate fatal exception in the console
        self._forcestop = False
        self._error_msg = ''

        # If available, use readline and rlcompleter to add functionality
        try:
            import readline
            self._readline = readline

            # Load any previous console command history
            try:                      readline.read_history_file()
            except FileNotFoundError: print('No history file found')

            # Follow history when executing prior commands (Windows only)
            self._enable_history_restoration()

            # Allow tab-completion of variables
            readline.parse_and_bind('tab: complete')
            readline.set_completer(self._patched_completer(variables))
        except ImportError:    pass

        # Any other errors should be logged once the console starts
        except: self._error_msg = f'\n{traceback.format_exc()}'


    def interact(self, banner: str | None = None, **kwargs):
        """ Start the console, ensuring command history written on exit """
        try: 
            super().interact(banner=f'{banner}{self._error_msg}', **kwargs)
        finally:
            if hasattr(self, '_readline'):
                try:                   self._readline.write_history_file()
                except Exception as e: print(f'Error writing history: {e}')

        # Raise a SystemExit if the forcestop flag was set while running
        if getattr(self, '_forcestop', False): raise SystemExit


    def raw_input(self, prompt: str = ''):
        """ Fixes issues with ANSI coloring and cursor location.

        Notes
        -----
        In some cases, default code.InteractiveConsole.raw_input function
        can misplace the cursor location when retrieving command history.

        In addition, pyreadline breaks RGB ANSI colors for input(). Both of
        these issues are fixed by printing the input prompt separately from
        the actual input call. Note that this fixes the console prompt, but 
        not ANSI coloring when actually using input().
        
        """
        
        # Remove all ANSI codes to get the printed length of the prompt
        # https://stackoverflow.com/a/68635860 
        remove_ansi = re.compile(r'[\u001B\u009B][\[\]()#;?]*'
            r'((([a-zA-Z\d]*(;[-a-zA-Z\d\/#&.:=?%@~_]*)*)?\u0007)'
            r'|((\d{1,4}(?:;\d{0,4})*)?[\dA-PR-TZcf-ntqry=><~]))')
        len_no_ansi = len(re.sub(remove_ansi, '', prompt))

        # Print the prompt, move the cursor back, use \x1b[nC to define new
        # prompt which is the same length but uses non-overwriting spaces
        print(prompt, end=''.join(['\b']*len_no_ansi))
        return super().raw_input(f'\x1b[{len_no_ansi}C')


    def _enable_history_restoration(self) -> bool:
        """ Restore location in history when executing previous commands.
        
        Notes
        -----
        Ordinarily when scrolling up in the command history and executing
        a command, the location in history will be reset to the end. This 
        means a user needs to again scroll back up to the same location if 
        they want to execute the command which followed the one that they 
        just executed.

        To address that inconvenience, this function enables restoring 
        history to the place it was at prior to executing the command - 
        thus allowing a user to immediately select successive commands from 
        history, without needing to re-navigate back through history after 
        each command. 
    
        Note that due to the dependency on pyreadline3, this functionality 
        is only available on Windows. While the core components that this 
        functionality depends on does exist in the underlying readline C 
        module, python's default readline library does not provide bindings 
        to them. With some effort, however, it should be possible to modify 
        this funtion so that it directly calls the readline.so module (and 
        binds to the proper interfaces, as pyreadline3 does in Windows).

        Returns
        -------
        bool
            True if it was successfully enabled, and False otherwise.

        """

        readline = getattr(self, '_readline', None)
        if not hasattr(readline, 'rl'): return False

        def restore_history(*args, **kwargs):
            """ Implements operate-and-get-next.

            Allow successive commands from history to be selected; i.e. 
            after navigating through the console's command history and 
            executing a command, the cursor will still be at the same 
            location in the command history so that the next relative 
            historical command can be fetched and edited.
            
            .. _operate-and-get-next reference:
                https://tiswww.case.edu/php/chet/readline/readline.html#index-operate_002dand_002dget_002dnext-_0028C_002do_0029

            """

            def _handle_exception(e):
                """ If there's an exception: set flag, print info, exit """
                self._forcestop = True
                print(f'\n{traceback.format_exc()}')
                print(f'Exception in restore_history: {e}')
                raise EOFError

            def _set_cursor(index, *args, **kwargs):
                """ Set the index and remove function from the queue """
                try:
                    terminal = readline.rl.mode
                    terminal._history.set_history_cursor(index)
                    terminal.process_keyevent_queue.pop(-1)
                    return terminal.process_keyevent(*args, **kwargs)
                except Exception as e: _handle_exception(e)

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
            except Exception as e: _handle_exception(e)

            # Otherwise just accept the line as usual
            return terminal.accept_line(*args, **kwargs)

        # Bind the restoration function to the enter key so
        # that it's run each time a command is executed
        readline.rl.mode._bind_key('return', restore_history)
        return True


    def _patched_completer(self, variables: dict):
        """ Fixes case where accessing an attribute can create an error """
        from rlcompleter import Completer
        from unittest.mock import patch

        class PatchedCompleter(Completer):
            def attr_matches(self, *args, **kwargs):
                """ Use unittest to wrap getattr in try/except """
                original_getattr = getattr
                def safe_getattr(*args, **kwargs):
                    try:    return original_getattr(*args, **kwargs)
                    except AttributeError: raise
                    except: return None
                with patch('builtins.getattr', safe_getattr):
                    return super().attr_matches(*args, **kwargs)
        return PatchedCompleter(variables).complete
