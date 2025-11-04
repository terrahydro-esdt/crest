from crest.model.Model import Model
from crest.archiver.Archiver import Archiver
from crest.data.batching.Batcher import Batcher
from crest.data.loading.Datafile import Datafile
from crest.data.loading.Dataset import Dataset
from crest.configuration.Config import Config
from crest.utils import Stopwatch
from functools import cached_property, partial

import logging
import os
import importlib
import pandas as pd
import numpy as np

from crest.model.ExtentStrategy import ExtentStrategy
from crest.utils.sys_metrics import SysMetrics

logger = logging.getLogger(__name__)
    
    
class GriddedModel():
    """ Handles the automatic generation of model predictions over multiple points based
        on region. 
    """

    def __init__(self, config: Config, alt_model_loader=None):
        logger.debug("GriddedModel: Starting __init__")

        self.sm = SysMetrics()
        self.config = config
        self.extent = self.config.extent
        self.region = self.config.region
        self.data_schema = None
        self.operation_data = []

        if hasattr(self.config, 'framework'):
            if (self.config.framework == 'crest'):
                self.is_base_node_type = False
            elif (self.config.framework == 'terrahydro'):
                self.is_base_node_type = True

        self.init_model_dataset()
        self.init_preprocessors()

        logger.debug("GriddedModel: Initializing extent strategy")
        self.extent_strategy = ExtentStrategy(config)

        logger.info('Load the model specified by the configuration.')
        self.models = self.load_model(alt_model_loader=alt_model_loader)
        logger.info('Initialized Gridded Model')

    @cached_property
    def benchmark(self):
        """ Return a Stopwatch function for benchmarking """
        return lambda label, logger=logger.debug, **kwargs: Stopwatch(**({
            'message' : f'GriddedModel.{label}',
            'logger'  : logger,
            'silent'  : {'time': 0.05}, # Don't log when time < 0.05 seconds
        } | kwargs))

    def _resolve_function_path(self, path: str):
        """ Resolves a function path to the actual function object. """

        logger.debug(f"Resolving function path: {path}")
        module_path, func_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def init_preprocessors(self):
        """ Initializes preprocessors based on the configuration. """

        logger.debug("Initializing preprocessors")
        self.preprocessors = {}

        if not hasattr(self.config, 'preprocessors') or not self.config.preprocessors:
            logger.debug("No preprocessors defined in config")
            return

        for key, fn_paths in self.config.preprocessors.items():
            try:
                logger.debug(f"Loading preprocessors for key: {key}")
                self.preprocessors[key] = {'preprocessors':
                    self._resolve_function_path(p) for p in fn_paths}
            except Exception as e:
                logger.exception(f"Could not load preprocessors for {key}")
                raise ImportError(
                    f"Could not load preprocessors for {key}: {e}")

    def init_model_dataset(self):
        """ Initializes the model dataset class if specified in the config. """

        logger.debug("Initializing model dataset class")

        if not hasattr(self.config, 'model_dataset_class') or not self.config.model_dataset_class:
            logger.debug("No model_dataset_class defined in config")
            return

        try:
            base_model_class = self._resolve_function_path(
                self.config.model_dataset_class)

            class ModelDataset(base_model_class):
                outputs = {}

            self.model_dataset_class = ModelDataset
            logger.debug("Model dataset class successfully created")

            logger.debug("Model_dataset_class =", self.model_dataset_class)
            logger.debug("Model type =", type(self.model_dataset_class))
        except (ImportError, AttributeError) as e:
            logger.exception(
                f"Error importing model_dataset_class '{self.config.model_dataset_class}'")
            raise ImportError(
                f"Could not import model_dataset_class '{self.config.model_dataset_class}': {e}")

    def load_model(self, alt_model_loader=None):
        """ Loads the model(s) specified in the configuration."""

        with self.benchmark(f'load_model'):

            logger.debug("Loading model(s)")
            models = []
            if isinstance(self.config.model_path, list):
                for i, model_path in enumerate(self.config.model_path):
                    model_type = self.config.model_type if not isinstance(
                        self.config.model_type, list) else self.config.model_type[i]
                    
                    model_version = self.config.model_version if not isinstance(
                        self.config.model_version, list) else self.config.model_version[i]
                    
                    model = self._load_single_model(
                        model_path, alt_model_loader=alt_model_loader, model_type=model_type)
                    
                    self.sm.record_model_info(name=model.name, path=model_path, version=model_version)
                    models.append(model)
            elif isinstance(self.config.model_path, str):
                model = self._load_single_model(
                    self.config.model_path, alt_model_loader=alt_model_loader, model_type=self.config.model_type)
                
                self.sm.record_model_info(name=model.name, path=self.config.model_path, version=self.config.model_version)
                models.append(model)
            else:
                logger.error("model_path in config has an invalid format")
                raise Exception(
                    'model_path in yaml config is in an unrecognized format.')

            logger.info(f"Loaded {len(models)} model(s)")
            return models
    
    def _load_single_model(self, model_path, alt_model_loader=None, model_type=None):
        """ Loads a single model from the specified path. """

        logger.debug(f"Loading single model: {model_path}")

        if alt_model_loader is None:
            model_type = model_type if model_type is not None else "keras"
            model = Model.load(model_path, model_type)
        else:
            loader = alt_model_loader(model_path)
            model = loader.load()

        return model
    
    def init_dataset(self, process_dataset: callable = None):
        try:
            logger.info("Updating Extents")
            self.extent = dict(self.extent_strategy.update_extents())
            logger.debug(f"{self.extent_strategy} new extent: {self.extent}")

            # Determine the targeted start and end datetimes
            target = self.config.process_dataset_args['target_dt']
            stamps = [pd.Timestamp(str(t)) for t in np.atleast_1d(target)]
            assert(len(stamps)<=2), f'Too many values: target_dt={target}'

            # Starting datetime is offset by the datetime lookback window
            # TODO: ending datetime should also be offset by the lookforward
            dt_depth = self.config.variable_depth['datetime'][0]
            ts_range = [stamps[0]-pd.Timedelta(hours=dt_depth), stamps[-1]]

            # Update the data extent and record the new datetime range
            dt_s,dt_e = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in ts_range]
            dt_extent = {'datetime' : [dt_s, dt_e]}
            self.sm.record_dataset_info(start_dt=dt_s, end_dt=dt_e)
            
            if not self.is_base_node_type:
                logger.info("Initializing dataset using Datafile list")
                self.data = Dataset([Datafile(**({
                    'region'       : self.region,
                    'extent'       : self.extent[source] | dt_extent,
                    'features'     : self.features[source],
                    'location'     : self.locations[source],
                    'window_depth' : self.depth[source],
                    } | self.preprocessors.get(source, {})
                )) for source in self.features])

            # Dataset handles its own creation if model inherits from BaseNode
            else:
                logger.debug('Calling Dataset.from_models with model class')
                self.extent |= dt_extent
                df_kwargs = {'*': {'extent': self.extent}}# | self.preprocessors
                self.data = Dataset.from_models(verbose=False, **{
                    'models': [self.model_dataset_class],
                    'database_folder': self.config.database_path,
                    'variable_depths': self.config.variable_depth,
                    'datafile_kwargs': df_kwargs,
                })

            if process_dataset is not None:
                logger.info("Applying process_dataset function")
                self.data = process_dataset(self.data, **self.config.process_dataset_args)

            # Schema should contain only the targeted datetime(s)
            if self.data_schema is None:
                logger.debug("Initializing data_schema from dataset")
                schema = self.data[self.config.data_schema_ind]
                schema = schema.data.to_dataset('features')
                self.data_schema = schema.sel(datetime=slice(*stamps))

            logger.debug(f'{self.data=}')
            logger.debug(f'{self.data_schema=}')

            logger.debug("Caching data")
            self.data.cache(**(self.config.cache_kwargs['train']))

            logger.info(f'Successfully initialized {self.data=}')
            return True
        except Exception as e:
            raise
            logger.exception(f'Dataset initialization failed. {e}')
            return False

    def predict(self):
        logger.info("Starting prediction phase")
        if hasattr(self, 'model_dataset_class') and self.model_dataset_class is not None:
            model_inputs = list(
                self.model_dataset_class.input_spec)
        else:
            model_inputs = self.config.inputs
    
        batch_predict = Batcher(self.data, **{
            'batch_size': self.config.batch_size,
            'repeat': self.config.repeat,
            'shuffle': self.config.shuffle,
            'numblocks': self.config.numblocks,
            'workers': self.config.workers,
            'seed': self.config.seed,
            'log_level': logging.DEBUG,
            'prequeuer' : partial(process_batch, model_inputs=model_inputs),
        })
        logger.info(f'Initialized batch_predict with size: {batch_predict.batch_size}')
        base_name = self.model.graph.name if not hasattr(self.config, 'output_name') else self.config.output_name
        output_dir = (os.path.join(self.config.output_path, base_name)) + '.zarr'
        logger.info(f'Outputting predictions to {output_dir}')


        try:
            with Archiver(self.data_schema, output_dir, overwrite=True, verbose=True) as a:
                with batch_predict as batcher:

                    for coord, value in batcher.generator(show_timing=True):
                        # with self.sm.record_batch(batch_idx=count, size=len(value[model_inputs[0]])):

                        pred = {}
                        for model in self.models:
                            pred = pred | model.predict_on_batch(value)
                        a.archive(coord, pred)

            # self.sm.finalize(True)
            return True
        except Exception as e:
            logger.exception('Could not complete predicting.')
            # self.sm.finalize(False)
            return False

import tlz
def process_batch(batch, model_inputs):
    extract = lambda v, dim: v.ravel()[-1 if dim == 'datetime' else v.size//2]
    keys = [('datetime',0), ('latitude',0), ('longitude',0)]
    coords = tlz.merge_with(np.array, 
        [{c : extract(s.coords[c][i], c) for c,i in keys} for s in batch])
    values = tlz.merge_with(np.array, [s.to_dict(model_inputs) for s in batch])
    return coords, values


# from crest.archiver.StageWriter import StageWriter
# def process_batch_full(model_inputs, loader, path, schema):       
#     model = loader(path).load()
#     archiver = StageWriter(schema, 'staging')

#     def process(batch):
#         coords, values = process_batch(batch, model_inputs)
#         pred = model.predict_on_batch(values)
#         archiver.stage_many_coords(coords, pred)
    
#     with archiver:
#         yield process
        