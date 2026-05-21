from __future__ import annotations
from collections.abc import Collection
from contextlib import contextmanager, redirect_stdout

import pandas as pd
import traceback
import io 


@contextmanager
def print_table(
    header  : Collection[str] = [], 
    sep     : str             = '|',
    options : dict            = {}, 
) -> None:
    """ Context manager to format printed text as a pandas table.
    
    Parameters
    ----------
    header  : Collection[str]
        Collection of strings to use as column labels. If an empty list
        is passed (default), then labels will use the format 'Column N'.
    sep     : str
        Separator to denote columns in the printed text (default '|').
    options : dict
        Any additional options that should be applied when printing the
        dataframe, passed into pd.option_context. For options, see here:
        pandas.pydata.org/docs/user_guide/options.html#available-options

    Examples
    --------
    >>> with print_table():
    ...   print('a | b | c')
    ...   print('d|e')
      Column 0 Column 1 Column 2
      -------- -------- --------
    0        a        b        c
    1        d        e
    >>> with print_table(['i*2','i*3','skip'], ' ', {'display.max_rows':4}):
    ...   for i in range(8):
    ...     print(f'{i} {i*2}')
       i*2 i*3
       --- ---
    0    0   0
    1    2   3
    ..  ..  ..
    6   12  18
    7   14  21

    """
    # Helper to print a list of items in the correct format
    def print_helper(*items): print(sep.join(map(str, items)))

    # First collect printed text within a StringIO buffer
    try:
        with redirect_stdout(io.StringIO()) as buffer: yield print_helper
    except: 
        print(buffer.getvalue())
        raise
        
    # Parse the rows / columns contained in the buffer
    size = lambda items: max(map(len, items))
    rows = buffer.getvalue().strip().split('\n')
    cols = [[r.strip() for r in row.split(sep)] for row in rows]

    try:
        # Create a dictionary using parsed columns and the header row
        ncols = size(cols)
        head  = list(header)[:ncols] 
        head += [f'Column {i}' for i in range(len(head), ncols)]
        head  = [(h, ''.join(['-']*len(h))) for h in head]
        table = {h: [row.pop(0) if row else '' for row in cols] for h in head}

        # No idea why pandas can't just accept a dictionary like a normal API
        with pd.option_context(*[o for opt in ({
            'display.max_rows'        : None,
            'display.max_columns'     : None,
            'display.show_dimensions' : False,
        } | options).items() for o in opt]):

            # Print the final DataFrame from the constructed table dictionary
            try:   print( pd.DataFrame(table) )
            except Exception as e:
                print(f'Exception: {e}\nTraceback:\n{traceback.format_exc()}')
                print('Failed to create table from the following dictionary:')
                for k,v in table.items(): print(f'{k:>{size(table)}} : {v}')
    
    except Exception as e:
        print(f'Exception: {e}\nTraceback:\n{traceback.format_exc()}')
        print('Failed to create table from the following lines:')
        for row in rows: print(row)