from pathlib import Path
import warnings
from dill import PicklingWarning,settings
from .model import HierarchalTensorGraph, Model, NetworkXGraph, Node, TensorSpec
from .data_server import DataServer
from .nodes import CrossStitchLSTM,LSTM,LSTMCell

ROOT_PATH = Path(__file__).parent


warnings.filterwarnings('ignore',category=PicklingWarning)
settings['byref'] = True

# from .utils.setup_logging import logger_setup
# logger_setup('crest-logfile')
