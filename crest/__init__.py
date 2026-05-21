from pathlib import Path
import warnings
try:
    from dill import PicklingWarning, settings as dill_settings
except ImportError:
    PicklingWarning = None
    dill_settings = {}

ROOT_PATH = Path(__file__).parent

# Ensure crest can be imported by spawned processed
import sys
sys.path.append(f'{ROOT_PATH.parent.as_posix()}')

import os, warnings
env_vars = {
    'MALLOC_TRIM_THRESHOLD_': {
        'value'  : '16384',
        'reason' : ' to fix a memory leak in dask: ' +
                   'https://github.com/dask/dask/issues/3530',
    },
    #'NUMBA_NUM_THREADS': {
    #    'value'  : '2',
    #    'reason' : ' to avoid over-saturating available resources',
    #},
}
for key, var in env_vars.items():
    if key not in os.environ:
        value, reason = var['value'], var.get('reason', '')
        # warnings.warn(f'Environment variable "{key}" is not set; ' +
        #               f'using a value of {value}{reason}', RuntimeWarning)
        os.environ[key] = value

# Batcher workers tend to core dump on external exit signals (e.g. keyboard
# interrupts), which can generate many multi-GB files. To prevent this, we
# just disable core dump files for our processes.
try:
    import resource
    if resource.getrlimit(resource.RLIMIT_CORE) != (0,0):
        # warnings.warn('Disabling core dumps to avoid generating many large ' +
        #               'files when background processes exit', RuntimeWarning)          
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError: pass

# TileDB and pyarrow both need to be imported prior to TensorFlow, or they segfault
try: import tiledb
except ImportError: pass
try: import pyarrow.parquet
except ImportError: pass
try: import pandas
except ImportError: pass

# from ._dask_monkeypatch import *
from .model import HierarchalTensorGraph, Model, NetworkXGraph, Node, TensorSpec,IOSpec
from .data_server import DataServer
try:
    from .nodes import CrossStitchLSTM,LSTM,LSTMCell
except: pass
if isinstance(PicklingWarning, type):
    warnings.filterwarnings('ignore', category=PicklingWarning)
    dill_settings['byref'] = True

# from .utils.setup_logging import logger_setup
# logger_setup('crest-logfile')
