import logging
import zarr
import xarray as xr
import importlib

logger = logging.getLogger(__name__)


class ExtentStrategy():

    def __init__(self, config):
        self.config = config

        self.extent_strategy_type = self.config.extent_strategy_type

        if self.extent_strategy_type == "steps":
            self.extent_strategy = self.get_datadiff_steps
            self.timesteps = self.config.timesteps
            self.last_time_ind = {k: None for k,
                                  v in self.config.features.items()}
            self.extent = {k: {} for k, v in self.config.features.items()}

        elif self.extent_strategy_type == "diff":
            self.extent_strategy = self.get_datadiff
            self.last_time_ind = {k: None for k,
                                  v in self.config.features.items()}
            self.extent = {k: {} for k, v in self.config.features.items()}

        elif self.extent_strategy_type == "fixed":
            self.extent_strategy = None
            self.last_time_ind = None

            self.extent = self.config.extent

        else:
            try:
                module_path, func_name = self.extent_strategy_type.rsplit(
                    ".", 1)
                module = importlib.import_module(module_path)
                self.extent_strategy = getattr(module, func_name)
            except Exception as e:
                raise ImportError(
                    f"Could not resolve extent strategy '{self.extent_strategy_type}': {e}")

    def update_extents(self):
        """ Iterate through sources and determine extent based on strategy defined."""

        try:

            if (self.extent_strategy_type in ["diff", "steps"]):
                for source in self.config.features.keys():
                    self.extent[source].update(self.extent_strategy(source))
            elif (self.extent_strategy_type == "fixed"):
                pass
            else:
                self.extent.update(self.extent_strategy())

            return self.extent
        except Exception as e:
            logger.exception(f'Unable to set extents for all sources. {e}')
            raise e

    def get_extent(self):
        return self.extent

    def get_datadiff(self, source: str) -> dict:
        """ Loads the datafile and creates a time-based sample of the datafile. 

        Parameters
        ----------
        source: str
            The string representation of the sources involved in the incoming dataset. 
        """

        # load datafile associated with source
        location = self.config.locations[source]
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

    def get_datadiff_steps(self, source: str) -> dict:
        """ Loads the datafile and creates a sample using the predetermined timestep param. 

        Parameters
        ----------
        source: str
            The string representation of the sources involved in the incoming dataset. 
        timesteps: int
            The number of timesteps we will be reading from the datafile at each iteration.
        """

        # load datafile from location[source]
        location = self.config.locations[source]
        timesteps = self.config.timesteps

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

            if (last_time_ind < (len(data.datetime.values) - 1)):
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
