from crest.model.Model import Model
from crest.utils.batcher_for_archiver import batch_for_archiver
from crest.archiver.Archiver import Archiver
from crest.data.batching.Batcher import Batcher
from crest.data.loading.Datafile import Datafile
from crest.data.loading.Dataset import Dataset
from crest.configuration.Config import Config
from crest.utils import Stopwatch
from functools import cached_property

from tqdm import tqdm
import logging
import os
import xarray as xr
import importlib
import pandas as pd
import json

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
                self.preprocessors[key] = [
                    self._resolve_function_path(p) for p in fn_paths]
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
            self.extent = self.extent_strategy.update_extents()
            logger.debug(f"Updated extent: {self.extent}")

            if not self.is_base_node_type:
                logger.info("Initializing dataset using Datafile list")
                
                self.data = Dataset([Datafile(
                    location=self.locations[source],
                    features=self.features[source],
                    window_depth=self.depth[source],
                    extent=self.extent[source],
                    region=self.region
                ) for source in self.features])
            else:
                target_dt = self.config.process_dataset_args['target_dt']

                if (isinstance(target_dt, str)):
                    start_dt = (pd.Timestamp(target_dt) - pd.Timedelta(hours=(self.config.variable_depth['datetime'][0] + 1))).strftime('%Y-%m-%d %H:%M:%S')
                    dt_range = {'datetime': [start_dt, pd.Timestamp(target_dt).strftime('%Y-%m-%d %H:%M:%S')]}
                else:
                    start_dt = (pd.Timestamp(target_dt[0]) - pd.Timedelta(hours=(self.config.variable_depth['datetime'][0] + 1))).strftime('%Y-%m-%d %H:%M:%S')
                    dt_range = {'datetime': [start_dt, pd.Timestamp(target_dt[-1]).strftime('%Y-%m-%d %H:%M:%S')]}

                logger.debug(f'{self.extent}=')

                self.sm.record_dataset_info(
                    start_dt=start_dt,
                    end_dt=target_dt if isinstance(target_dt, str) else target_dt[-1]
                )

                self.extent = self.extent | dt_range
                # self.extent = dt_range

                print(f'{self.extent=}')

                self.sm.record_dataset_info(
                    start_dt=start_dt,
                    end_dt=target_dt if isinstance(target_dt, str) else target_dt[-1]
                )

                datafile_kwargs = {'*': {'extent': self.extent}}
                for key, funcs in self.preprocessors.items():
                    datafile_kwargs[key] = {'preprocessors': funcs}

                logger.info('Datafile arguments: ' + str(datafile_kwargs))
                logger.debug("Calling Dataset.from_models with model class")

                self.data = Dataset.from_models(verbose=False, **{
                    'models': [self.model_dataset_class],
                    'database_folder': self.config.database_path,
                    'variable_depths': self.config.variable_depth,
                    'datafile_kwargs': datafile_kwargs,
                })

            if process_dataset is not None:
                logger.info("Applying process_dataset function")
                self.data = process_dataset(self.data, **self.config.process_dataset_args)

            if self.data_schema is None:
                logger.debug("Initializing data_schema from dataset")
                self.data_schema = {
                    'datetime': self.data[self.config.data_schema_ind]._raw_data['datetime'],
                    'latitude': self.data[self.config.data_schema_ind].data['latitude'],
                    'longitude': self.data[self.config.data_schema_ind].data['longitude']
                }

            logger.debug(f'{self.data=}')
            logger.debug(f'{self.data_schema=}')

            logger.debug("Caching data")
            self.data.cache(**(self.config.cache_kwargs['train']))

            logger.info(f'Successfully initialized dataset with {self.region}')
            return True
        except Exception as e:
            logger.exception(f'Dataset initialization failed. {e}')
            return False

    def predict(self):
        logger.info("Starting prediction phase")

        batch_predict = Batcher(self.data, **{
            'batch_size': self.config.batch_size,
            'repeat': self.config.repeat,
            'shuffle': self.config.shuffle,
            'numblocks': self.config.numblocks,
            'workers': self.config.workers,
            'seed': self.config.seed,
            'log_level': logging.DEBUG
        })

        logger.info(f'Initialized batch_predict with size: {batch_predict.batch_size}')

        base_name = self.model.graph.name if not hasattr(self.config, 'output_name') else self.config.output_name
        output_dir = (os.path.join(self.config.output_path, base_name)) + '.zarr'

        logger.info(f'Outputting predictions to {output_dir}')

        # a = Archiver(output_path=output_dir,
        #                 data_schema=self.data_schema,
        #                 datafile_index=self.config.coordinate_df,
        #                 # task_bytes=1e9,
        #                 overwrite=True)

        try:
            # with a, batch_predict as batcher:
            with batch_predict as batcher:
                logger.debug("Entered batcher context")

                if hasattr(self, 'model_dataset_class') and self.model_dataset_class is not None:
                    model_inputs = list(
                        self.model_dataset_class.input_spec)
                else:
                    model_inputs = self.config.inputs

                batcher_archiver = batch_for_archiver(batcher,
                                                    model_inputs=model_inputs,
                                                    coordinate_df=self.config.coordinate_df,
                                                    dataset=self.data)
            
                count = 0
                for coord, value in tqdm(batcher_archiver):
                    # with self.sm.record_batch(batch_idx=count, size=len(value[model_inputs[0]])):
                    logger.debug(f"{coord=}")

                    if (count == 5):
                        break
                        # pred = {}
                        # for model in self.models:
                        #     pred = pred | model.predict_on_batch(value)

                        # a.archive(coord, pred)
                    count += 1

            logger.info(f'Successfully completed generating and archiving predictions: {count} batches processed.')
            # self.sm.finalize(True)
            return True
        except Exception as e:
            logger.exception('Could not complete predicting.')
            # self.sm.finalize(False)
            return False