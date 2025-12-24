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
from crest.archiver.Archiver import Archiver

from tqdm import tqdm
import toolz as tlz
import logging
import os
import numpy as np
import xarray as xr
import importlib
from contextlib import contextmanager
from crest.archiver.StageWriter import StageWriter
from functools import partial

from crest.model.ExtentStrategy import ExtentStrategy
from crest.utils.sys_metrics import SysMetrics

logger = logging.getLogger(__name__)


def process_batch(batch, model_inputs):
    def extract(v, dim): return v.ravel(
    )[-1 if dim == 'datetime' else v.size//2]
    keys = [('datetime', 0), ('latitude', 0), ('longitude', 0)]
    coords = tlz.merge_with(np.array,
                            [{c: extract(s.coords[c][i], c) for c, i in keys} for s in batch])
    values = tlz.merge_with(
        np.array, [s.to_dict(model_inputs) for s in batch])
    
    if 'datetime' in values.keys(): 
        values['datetime'] = values['datetime'].astype('float32')
    return coords, values


@contextmanager
def process_batch_full(model_name, model_inputs, loader, schema, staging_dir):
    model = loader(model_name).load()

    # Get a Dataset object if a DataArray with a features dim was given
    if isinstance(schema, xr.DataArray):
        if 'features' in schema.dims:
            schema = schema.to_dataset('features')
    # Get a DataArray object if a Dataset was given
    if isinstance(schema, xr.Dataset):
        schema = schema[list(schema)[0]]

    archiver = StageWriter(schema, staging_dir)

    def process(batch):
        coords, values = process_batch(batch, model_inputs)
        pred = model.predict_on_batch(values)
        archiver.stage_many_coords(coords, pred)
    with archiver:
        yield process


class GriddedModel():
    """ Handles the automatic generation of model predictions over multiple points based
        on region. 
    """

    def __init__(self, config: Config, alt_model_loader=None, process_model: dict = {}):
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

        self.preprocess_model = process_model.get('preprocess', None)
        self.postprocess_model = process_model.get('postprocess', None)

        logger.debug("GriddedModel: Initializing extent strategy")
        self.extent_strategy = ExtentStrategy(config)

        self.alt_model_loader = alt_model_loader

        # logger.info('Load the model specified by the configuration.')
        # self.models = self.load_model(alt_model_loader=alt_model_loader)
        # logger.info('Initialized Gridded Model')

    def _resolve_function_path(self, path: str):
        """ Resolves a function path to the actual function object. """

        logger.debug(f"Resolving function path: {path}")
        module_path, func_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def init_dataset(self, process_dataset: dict = {}):
        try:
            logger.info("Updating Extents")
            self.extent = dict(self.extent_strategy.update_extents())
            logger.debug(f"{self.extent_strategy} new extent: {self.extent}")

            # Determine the targeted start and end datetimes
            target = self.config.target_dt
            stamps = [pd.Timestamp(str(t)) for t in np.atleast_1d(target)]
            assert (len(stamps) <= 2), f'Too many values: target_dt={target}'

            # Starting datetime is offset by the datetime lookback window
            # TODO: ending datetime should also be offset by the lookforward
            dt_depth = self.config.variable_depth['datetime'][0]
            ts_range = [stamps[0]-pd.Timedelta(hours=dt_depth), stamps[-1]]

            # Update the data extent and record the new datetime range
            dt_s, dt_e = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in ts_range]
            dt_extent = {'datetime': [dt_s, dt_e]}
            self.sm.record_dataset_info(start_dt=dt_s, end_dt=dt_e)

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
                if process_dataset and ('preprocess' in process_dataset):
                    logger.info("Applying process_dataset function")
                    datafile_kwargs = process_dataset['preprocess'](
                        **self.config.preprocess_args)
                else:
                    datafile_kwargs = {'*': {'extent': self.config.extent}}

                logger.info('Datafile arguments: ' + str(datafile_kwargs))
                logger.debug("Calling Dataset.from_models with model class")

                base_model_class = self._resolve_function_path(
                    self.config.model_dataset_class)

                if self.preprocess_model is not None:
                    self.model_dataset_class = self.preprocess_model(
                        base_model_class)
                else:
                    self.model_dataset_class = base_model_class

                self.data = Dataset.from_models(verbose=False, **{
                    'models': [self.model_dataset_class],
                    'database_folder': self.config.database_path,
                    'variable_depths': self.config.variable_depth,
                    'datafile_kwargs': datafile_kwargs,
                })

            if process_dataset and 'postprocess' in process_dataset:
                logger.info("Applying process_dataset function")
                self.data = process_dataset['postprocess'](
                    self.data, **self.config.postprocess_args)

            logger.debug("Caching data")
            self.data.cache(**(self.config.cache_kwargs['train']))

            for df in self.data:
                logger.info(
                    f'Datafile: {df.data.to_dataset("features")["latitude"]}')

            logger.info(f'Successfully initialized dataset with {self.region}')
            return self.data
        except Exception as e:
            logger.exception(f'Dataset initialization failed. {e}')
            return None

    def predict(self, data_schema):
        logger.info("Starting prediction phase")

        loader = self.alt_model_loader(self.config.model_path)
        model = loader.load()

        batch_predict = Batcher(self.data, **{
            'batch_size': self.config.batch_size,
            'repeat': self.config.repeat,
            'shuffle': self.config.shuffle,
            'numblocks': self.config.numblocks,
            'workers': self.config.workers,
            'seed': self.config.seed,
            'log_level': logging.DEBUG,
            'log_file': self.config.prediction_log,
            # 'prequeuer'      : partial(process_batch,
            #                            model_inputs=list(ifs_model.graph.inputs.keys())),
            'prequeuer': partial(process_batch_full,
                                 model_name=self.config.model_path,
                                 model_inputs=list(model.graph.inputs.keys()),
                                #  path=f'{loader.base_model.__name__}_cache',
                                 loader=self.alt_model_loader,
                                 schema=data_schema,
                                 staging_dir=self.config.archive_kwargs['staging_dir']
                                 ),
        })

        logger.info(
            f'Initialized batch_predict with size: {batch_predict.batch_size}')

        base_name = self.model.graph.name if not hasattr(
            self.config, 'output_name') else self.config.archive_kwargs['output_name']
        output_dir = (os.path.join(
            self.config.archive_kwargs['output_path'], base_name)) + '.zarr'

        logger.info(f'Outputting predictions to {output_dir}')
        try:
            with Archiver(data_schema=data_schema,
                          output_path=output_dir,
                          stage_path=self.config.archive_kwargs['staging_dir'],
                          overwrite=self.config.archive_kwargs['overwrite'],
                          verbose=self.config.archive_kwargs['verbose']) as a:
                with batch_predict as batcher:
                    for i, _ in enumerate(batcher.generator(show_timing=True)):
                        if (i % 1000) == 0:
                            a.zarr_writer.flush()

                logger.info(
                    f'Successfully completed generating and archiving predictions.')
                return True
        except Exception as e:
            logger.exception('Could not complete predicting.')
            return False
