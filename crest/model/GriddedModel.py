from functools import partial
from contextlib import contextmanager
import xarray as xr
import logging
import toolz as tlz
import numpy as np
import pandas as pd
import shutil
import importlib
import os
from crest.configuration.Config import Config
from crest.data.loading.Dataset import Dataset
from crest.data.loading.Datafile import Datafile
from crest.data.batching.Batcher import Batcher
from crest.archiver.Archiver import Archiver
from crest.archiver.StageWriter import StageWriter
from crest.utils.sys_metrics import SysMetrics
from crest.model.ExtentStrategy import ExtentStrategy
from tensorflow.python.framework import ops
import tensorflow as tf
from dask.diagnostics import ProgressBar
import dask
import pickle
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger(__name__)

def process_batch(batch, model_inputs, worker_id):
    def extract(v, dim): return v.ravel(
    )[-1 if dim == 'datetime' else v.size//2]
    keys = [('datetime', 0), ('latitude', 0), ('longitude', 0)]
    coords = tlz.merge_with(np.array,
                            [{c: extract(s.coords[c][i], c) for c, i in keys} for s in batch])
    values = tlz.merge_with(np.array, [s.to_dict(model_inputs) for s in batch])
    if 'datetime' in values.keys():
        values['datetime'] = values['datetime'].astype('float32')
    values = {k: ops.EagerTensor(v, f'/device:CPU:{worker_id}', None)
              for k, v in values.items()}
    return coords, values


@contextmanager
def process_batch_full(batcher, model_inputs, loader, schema, model_name, staging_dir):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    physical_devices = tf.config.list_physical_devices('CPU')
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration()
         for _ in range(batcher.workers)])

    logical_devices = tf.config.list_logical_devices('CPU')
    assert len(logical_devices) == batcher.workers
    worker_id = batcher.pidx
    with tf.device(f'/device:CPU:{worker_id}'):
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
        with tf.device(f'/device:CPU:{worker_id}'):
            coords, values = process_batch(batch, model_inputs, worker_id)
            pred = model.predict_on_batch(values)
            archiver.stage_many_coords(coords, pred)
    with archiver:
        yield process


class GriddedModel():
    """ Handles the automatic generation of model predictions over multiple points based
        on region. 
    """

    def __init__(self, config: Config, database_path=None, alt_model_loader=None, process_model: dict = {}, process_output: dict = {}, benchmarking: dict = {}):
        logger.debug("GriddedModel: Starting __init__")

        self.sm = SysMetrics(run_id="gridded-model", run_dir='/ASTG/sw/kraken')
        self.config = config
        self.extent = self.config.extent
        self.region = self.config.region
        self.data_schema = None
        self.database_path = database_path
        self.operation_data = []

        if hasattr(self.config, 'framework'):
            if (self.config.framework == 'crest'):
                self.is_base_node_type = False
            elif (self.config.framework == 'terrahydro'):
                self.is_base_node_type = True

        self.preprocess_model = process_model.get('preprocess', None)
        self.postprocess_model = process_model.get('postprocess', None)

        self.postprocess_output = process_output.get('postprocess', None)

        self.benchmarking = benchmarking

        logger.debug("GriddedModel: Initializing extent strategy")
        self.extent_strategy = ExtentStrategy(config)

        self.alt_model_loader = alt_model_loader

        self.sm.emit("model_info",
            model_name=getattr(self.config, "model_dataset_class", "unknown"),
            model_path=getattr(self.config, "model_path", "unknown"),
            model_version=getattr(self.config, "model_version", None)
        )
        self.sm.emit("dataset_info",
            region=getattr(self.config, "region", None),
            extent=str(getattr(self.config, "extent", None)),
            n_workers=getattr(self.config, "workers", None),
            variable_depth=getattr(self.config, "variable_depth", None)
        )

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

            with self.sm.timed("update_extents"):
                self.extent = dict(self.extent_strategy.update_extents())
                logger.debug(f"{self.extent_strategy} new extent: {self.extent}")

            # Determine the targeted start and end datetimes
            with self.sm.timed("resolve_datetime_range"):
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

                self.sm.emit("datetime_range", start_dt=dt_s, end_dt=dt_e)

            with self.sm.timed("dataset_build"):
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

            # ADD dataset benchmarking 
            benchmark_tdt = self.benchmarking.get('benchmark_tdt', None)
            benchmark_tdt(self.data, self.database_path)

            logger.debug("Caching data")
            with self.sm.timed("dataset_caching"):
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

        with self.sm.timed("model_load"):
            loader = self.alt_model_loader(self.config.model_path)
            model = loader.load()

            self.sm.emit("loaded_model",
                model_name=getattr(model, "name", "tf_model"),
                model_path=self.config.model_path,
                model_version=getattr(self.config, "model_version", None),
            )

        with self.sm.timed("batcher_init"):
            batch_predict = Batcher(self.data, **{
                'batch_size': self.config.batch_size,
                'repeat': self.config.repeat,
                'shuffle': self.config.shuffle,
                'numblocks': self.config.numblocks,
                'workers': self.config.workers,
                'seed': self.config.seed,
                'log_level': logging.DEBUG,
                'log_file': self.config.prediction_log,
                'task_bytes': self.config.task_bytes,
                # 'prequeuer'      : partial(process_batch,
                #                            model_inputs=list(ifs_model.graph.inputs.keys())),
                'prequeuer': partial(process_batch_full,
                                    model_inputs=list(model.graph.inputs.keys()),
                                    loader=self.alt_model_loader,
                                    schema=data_schema,
                                    model_name=self.config.model_path,
                                    staging_dir=self.config.archive_kwargs['staging_dir']
                                    ),
            })

        logger.info(
            f'Initialized batch_predict with size: {batch_predict.batch_size}')

        try:
            with self.sm.timed("archiver_open"):
                base_name = model.graph.name if not hasattr(
                    self.config, 'output_name') else self.config.archive_kwargs['output_name']
                output_path = (os.path.join(
                    self.config.archive_kwargs['output_path'], base_name)) + '.zarr'

            logger.info(f'Outputting predictions to {output_path}')
            with Archiver(data_schema=data_schema,
                          output_path=output_path,
                          stage_path=self.config.archive_kwargs['staging_dir'],
                          overwrite=self.config.archive_kwargs['overwrite'],
                          verbose=self.config.archive_kwargs['verbose']) as a:
                with batch_predict as batcher:
                    self.sm.emit("record_dataset_info",
                        n_workers=batcher.workers,
                        n_rows=getattr(self.config, "batch_size", None),
                    )

                    with self.sm.timed("predict_loop"):
                        for i, _ in enumerate(batcher.generator(show_timing=sys.stderr.isatty())):
                            if (i % 2000) == 0:
                                a.zarr_writer.flush()

                logger.info(
                    f'Successfully completed generating and archiving predictions.')
                logger.info(f"Started output postprocess")

            with self.sm.timed("postprocess_output"):
                with dask.config.set(scheduler="synchronous", num_workers=24):
                    forcing = self.data[0].data.to_dataset('features')
                    mask = forcing[self.config.ocean_mask]
                    temp = forcing[self.config.temperature]
                    temp = temp.where(mask>=0.5)

                    if (self.config.temperature_unit == 'K'):
                        temp = temp - 273.15
                    elif (not self.config.temperature_unit == 'C'):
                        raise ValueError('Unknown Temperature Unit')

                    mask = mask.where(mask>=0.5).isel({'datetime': -1}).drop_vars(('datetime')).compute()

                    # pickle_path = os.path.join(self.config.archive_kwargs['output_path'], f"{base_name}_post_process.pkl")
                    # with open(pickle_path, 'wb') as f:
                    #     pickle.dump((mask, temp), f)
                    # logger.info(f"Saved postprocess inputs to {pickle_path}")

                    post_process_output = (os.path.join(
                        self.config.archive_kwargs['output_path'], 
                        f"{base_name}_post_process")) + '.zarr'

                    data=self.postprocess_output(output_path, mask, temp, **self.config.postprocess_output_args)
                    with ProgressBar():
                        data.chunk('auto').to_zarr(post_process_output, mode='w', consolidated=True, align_chunks=True)

                shutil.rmtree(output_path)
                os.rename(post_process_output, output_path)
                logger.info(f"Renamed postprocessed output")

                logger.info(f"Completed output postprocess.")
                self.sm.emit("run_complete", status="success")

            return True
        except Exception as e:
            logger.exception(f'Could not complete predicting. {e}')
            self.sm.emit("run_complete", status="failed", error=str(e))

            # TODO Change this back
            return True
