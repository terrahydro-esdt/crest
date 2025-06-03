from .Model import Model
from ..utils.batcher_for_archiver import batch_for_archiver
from ..archiver.Archiver import Archiver
from ..data.batching.Batcher import Batcher
from ..data.loading.Datafile import Datafile
from ..data.loading.Dataset import Dataset
from ..configuration.Config import Config

from tqdm import tqdm
import logging
import os
import xarray as xr
import zarr

logger = logging.getLogger(__name__)


class GriddedModel():
    """ Handles the automatic generation of model predictions over multiple points based
        on region. 
    """

    def __init__(self, config: Config):
        """ Initializes the Gridded Model class. 

        Parameters
        ----------
        config: CREST Config
            This object contains an underlying dictionary of parameters necessary to initialize
            DataFile, DataSet, and Model. Initialization will fail otherwise. 
        """

        self.config = config
        self.features = self.config.features
        self.locations = self.config.locations
        self.depth = self.config.depth
        self.extent = self.config.extent
        self.region = self.config.region
        self.count = 0
        self.last_time_ind = {k: None for k, v in self.features.items()}
        self.timesteps = self.config.timesteps
        self.data_schema = None

        logger.info(f'Load the mode specified by the configuration.')
        self.model = self._load_model()

        logger.info(f'Working on model {self.model.graph.name}')

        logger.info(
            f'Initialized Gridded Model')

    def get_datadiff(self, source: str) -> dict:
        """ Loads the datafile and creates a time-based sample of the datafile. 

        Parameters
        ----------
        source: str
            The string representation of the sources involved in the incoming dataset. 
        """

        # load datafile associated with source
        location = self.locations[source]
        location = zarr.DirectoryStore(location)
        data = xr.open_zarr(location)

        # if the datetime can't be sampled then return empty
        if "datetime" not in data.coords:
            return {}

        # get the index of last time step in the source
        last_time_ind = self.last_time_ind[source]
        if (last_time_ind is None):
            last_time_ind = 0

        # get the previous and next datetime with which we will sample
        last_time = (data.datetime.values)[last_time_ind]
        current_time = (data.datetime.values)[-1]

        # if the times are the same, there are no samples
        if (current_time == last_time):
            logger.error('Time steps are not different.')
            return {}

        # set extent
        extent = {"datetime": [last_time, current_time]}

        # set the current time as the last time
        last_time_ind = len(data.datetime.values) - 1
        self.last_time_ind[source] = last_time_ind

        return extent

    def get_datadiff_steps(self, source: str, timesteps: int) -> dict:
        """ Loads the datafile and creates a sample using the predetermined timestep param. 

        Parameters
        ----------
        source: str
            The string representation of the sources involved in the incoming dataset. 
        timesteps: int
            The number of timesteps we will be reading from the datafile at each iteration.
        """

        # load datafile from location[source]
        location = self.locations[source]

        logger.info(f'Defining extents with {location}')

        location = zarr.DirectoryStore(location)
        data = xr.open_zarr(location)

        # if we can't sample with datetime, return empty
        if "datetime" not in data.coords:
            return {}

        # extent can't be set if timesteps are invalid
        if timesteps <= 0 or timesteps >= data.datetime.size:
            logger.error(f"Invalid number of timesteps: {timesteps}")
            return {}

        # get previous index
        last_time_ind = self.last_time_ind[source]
        if (last_time_ind is None):
            last_time_ind = 0

        current_time_ind = last_time_ind + timesteps
        if (current_time_ind >= len(data.datetime.values)):
            logger.error(f'Invalid window size based on previous index')

            if (last_time_ind < (len(data.datetime.value) - 1)):
                current_time_ind = -1
            else:
                return {}

        # set latest sample window based on timestep
        last_time = (data.datetime.values)[last_time_ind]
        current_time = (data.datetime.values)[current_time_ind]

        if (current_time == last_time):
            logger.error('Time steps are not different.')
            return {}

        extent = {"datetime": [last_time, current_time]}

        self.last_time_ind[source] = current_time_ind

        return extent

    def get_ts_extents(self):
        """ Iterate through sources and determine extent based on strategy defined."""

        try:
            for source in self.features.keys():
                if (self.timesteps is None):
                    self.extent[source].update(self.get_datadiff(source))
                else:
                    self.extent[source].update(
                        self.get_datadiff_steps(source, self.timesteps))
        except Exception as e:
            logger.exception(f'Unable to set extents for all sources. {e}')
            raise e

    def init_dataset(self):
        """ Initialize the dataset object and cache the data. """
        try:
            logger.info('Updating Extents.')

            self.get_ts_extents()

            logger.debug(f'Extent: {self.extent}')

            logger.info(f'Initializing dataset')

            self.data = Dataset([Datafile(
                location=self.locations[source],
                features=self.features[source],
                window_depth=self.depth[source],
                extent=self.extent[source],
                region=self.region
            ) for source in self.features])

            # this only works when the model extends basenode
            # logger.info(self.model.graph.node)

            # self.data = Dataset.from_models(verbose=False, **{
            # 'models'     : [self.model],
            # 'database_folder' : Path(self.config.database_path),
            # 'variable_depths' : self.config.variable_depth,
            # 'datafile_kwargs' : {
            #     '*': self.config.extent },
            # })

            # alternate extent definition
            # """
            # 'datafile_kwargs' : {
            #   '*': {'extent': extent[k]},
            #   'Topography' : {'preprocessors': [coarsen(9)]},
            # """

            if (self.data_schema is None):
                self.data_schema = {
                    'datetime': self.data[self.config.data_schema_ind]._raw_data['datetime'],
                    'latitude': self.data[self.config.data_schema_ind].data['latitude'],
                    'longitude': self.data[self.config.data_schema_ind].data['longitude']
                }

            self.data.cache(**(self.config.cache_kwargs['train']))

            logger.info(f'Successfully initialized dataset with {self.region}')
            return True
        except Exception as e:
            logger.exception(f'Dataset initialization failed. {e}')
            return False

    def _load_model(self):
        """ Loading the model. """
        logging.info(f'Load Model')

        model = Model.load(self.config.model_path, self.config.model_type)
        return model

    def predict(self):
        """ Generates predictions for each of the batche partitions of the DataFile. 
            Automatically archives these predictions as a means to combine the results.     
        """

        try:
            batch_predict = Batcher(self.data, **{
                'batch_size': self.config.batch_size,
                'repeat': self.config.repeat,
                'shuffle': self.config.shuffle,
                'numblocks': self.config.numblocks,
                'workers': self.config.workers,
                'seed': self.config.seed
            })

            logger.info(
                f'Initialized batch_predict with size: {batch_predict.batch_size}')

            base_name = self.model.graph.name if not hasattr(
                self.config, 'output_name') else self.config.output_name

            output_dir = (os.path.join(
                self.config.output_path, base_name)) + '.zarr'

            logger.info(f'Outputting predictions to {output_dir}')

            if (os.path.exists(output_dir)):
                output_data = xr.open_zarr(output_dir)
                print('output_data', output_data.coords)

            logger.info(f'data_schema is initialized with first row of data')

            with Archiver(output_path=output_dir,
                          data_schema=self.data_schema,
                          datafile_index=self.config.coordinate_df,
                          overwrite=False) as a:

                with batch_predict as batcher:

                    batcher_archiver = batch_for_archiver(batcher,
                                                        model_inputs = self.config.inputs,
                                                        coordinate_df = self.config.coordinate_df,
                                                        dataset=self.data)
                    for coord, value in tqdm(batcher_archiver):

                        pred = self.model.predict_on_batch(value)
                        a.archive(coord, pred)

            logger.info(
                f'Successfully completed generating and archving predictions for iter {self.count=}')

            return True
        except Exception as e:
            logger.exception('Could not complete predicting.')
            return False
