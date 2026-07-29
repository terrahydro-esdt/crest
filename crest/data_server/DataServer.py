""" 
This module implements a data server that downloads
remote data sources. It's intended to be used to
help set up examples and run tests that require
data to run.
"""

import logging
import os
import pickle as pkl
from urllib.request import urlretrieve

import xarray as xr

from crest.data.loading.StructuredDataset import StructuredDataset

logger = logging.getLogger(__name__)


class DataServer:
    """ Data Server for loading files from crest server"""

    @staticmethod
    def _download_and_convert_zarr(dataset_name, url, path, files_config):
        """
        Helper method to download NetCDF files and convert to Zarr format.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset being downloaded (for logging)
        url : str
            Base URL for downloading files
        path : str
            Local path to store the dataset
        files_config : list of dict
            List of file configurations with keys:
            - 'netcdf': NetCDF filename
            - 'zarr': Zarr path/filename
            - 'zarr_processor': Optional function to process zarr conversion

        Returns
        -------
        str
            Path to the downloaded dataset, or None on failure
        """
        logger.debug(f'Loading dataset: {dataset_name}')
        logger.debug(f'Source URL: {url}')
        logger.debug(f'Target path: {path}')

        # Make directory if it doesn't exist
        if not os.path.isdir(path):
            os.mkdir(path)
            logger.debug(f'Created directory: {path}')

        for config in files_config:
            netcdf_name = config['netcdf']
            zarr_name = config['zarr']
            zarr_processor = config.get('zarr_processor', None)

            file = os.path.join(path, netcdf_name)
            file_zarr = os.path.join(path, zarr_name)

            if not os.path.isfile(file) and not os.path.exists(file_zarr):
                logger.debug(f'Downloading {netcdf_name} from {url}')
                urlretrieve(url + netcdf_name, file)

                # Convert files to zarr format
                logger.debug(f'Converting {netcdf_name} to Zarr format')
                with xr.open_dataset(file) as data:
                    if zarr_processor:
                        zarr_processor(data, path, netcdf_name)
                    else:
                        data.to_zarr(os.path.join(path, netcdf_name.split('.')[0] + '.zarr'))

                # Remove the netcdf files
                os.remove(file)
                logger.debug(f'Removed temporary NetCDF file: {netcdf_name}')
            else:
                logger.debug(f'File already exists: {zarr_name}')

        logger.info(f'Download complete for {dataset_name}')
        return path

    @staticmethod
    def load_mnist(path=''):
        """
        Creates a folder /mnist and stores MNIST
        data for examples/mnist_demo.ipynb. If the folder
        already exist, it reads from the local files, otherwise
        it downloads it from the server.

        Parameters
        ----------

        path, optional : Local path to store MNIST. If not
        specified, it stores it in crest/data_server

        Returns
        -------

        List with StructuredDatasets [train,valid,test,prediction]

        """

        # MNISTs StructuredDatasets
        data = []

        # URL to mnist data on server
        url = 'https://portal.nccs.nasa.gov/datashare/astg/crest/examples/'

        # Default path
        if not path:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist')
        else:
            path = os.path.join(path, 'mnist')

        # Make directory if it doesn't exist
        if not os.path.isdir(path):
            os.mkdir(path)

        # Load files into StructuredDatasets
        for i in ['train', 'valid', 'test', 'pred']:
            name = 'mnist_' + i + '.json'
            file = os.path.join(path, name)

            # Download from server if needed
            if not os.path.isfile(file):
                urlretrieve(url + name, file)

            # Create StructuredDatasets
            with open(file, 'r') as f:
                json_string = f.read()
                data.append(StructuredDataset.from_json(json_string))

        return data
    
    @staticmethod
    def load_streamflow(path=''):
        """
        Creates a folder /streamflow and stores streamflow
        data for examples/streamflow_demo.ipynb. If the folder
        already exist, it reads from the local files, otherwise
        it downloads it from the server.

        Parameters
        ----------

        path, optional : Local path to store streamflow. If not
        specified, it stores it in crest/data_server

        Returns
        -------

        List with StructuredDatasets [train,valid,test,prediction]

        """

        # Streamflow StructuredDatasets
        data = []

        # URL to streamflow data on server
        url = 'https://portal.nccs.nasa.gov/datashare/astg/crest/examples/'

        # Default path
        if not path:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lstm_streamflow')
        else:
            path = os.path.join(path, 'lstm_streamflow')

        # Make directory if it doesn't exist
        if not os.path.isdir(path):
            os.mkdir(path)

        # Load files into StructuredDatasets
        name = 'lstm_streamflow.pkl'
        file = os.path.join(path, name)

        # Download from server if needed
        if not os.path.isfile(file):
            urlretrieve(url + name, file)

        # Create StructuredDatasets
        with open(file, 'rb') as f:
            pkl_data = pkl.load(f)
            data = xr.Dataset.from_dict(pkl_data)

        return data

    @staticmethod
    def load_soil_moisture(path=''):
        """
        Creates a folder /soil_moisture and stores required
        data for examples/soil_moisture_demo.ipynb. If the folder
        already exist, it reads from the local files, otherwise
        it downloads it from the server.

        Parameters
        ----------

        path, optional : Local path to store soil_moisture. If not
        specified, it stores it in terrahydro/toy_dataset

        Returns
        -------

        Path to the downloaded dataset

        """
        # URL to the soil_moisture data on server
        url = 'https://portal.nccs.nasa.gov/datashare/astg/terrahydro/toy_dataset/'

        # Default path
        if not path:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'soil_moisture')
        else:
            path = os.path.join(path, 'soil_moisture')

        def soil_moisture_processor(data, base_path, filename):
            """ Custom processor for soil moisture files """
            if filename in ['SMAP.nc', 'ERA5.nc']:
                data.to_zarr(os.path.join(base_path, filename.split('.')[0] + '.zarr'))
            else:
                data.to_zarr(os.path.join(base_path, 'StaticAttributes.zarr/' + filename.split('.')[0]))

        # Configuration for files to download
        files_config = [
            {'netcdf': 'SMAP.nc', 'zarr': 'SMAP.zarr', 'zarr_processor': soil_moisture_processor},
            {'netcdf': 'ERA5.nc', 'zarr': 'ERA5.zarr', 'zarr_processor': soil_moisture_processor},
            {'netcdf': 'Soil.nc', 'zarr': 'StaticAttributes.zarr/Soil', 'zarr_processor': soil_moisture_processor},
            {'netcdf': 'Irrigation.nc', 'zarr': 'StaticAttributes.zarr/Irrigation', 'zarr_processor': soil_moisture_processor}
        ]

        return DataServer._download_and_convert_zarr('soil_moisture', url, path, files_config)

    @staticmethod
    def load_evapotranspiration(path=''):
        """
        Creates a folder /evapotranspiration and stores required
        data for examples/evapotranspiration_demo.ipynb. If the folder
        already exist, it reads from the local files, otherwise
        it downloads it from the server.

        Parameters
        ----------

        path, optional : Local path to store evapotranspiration. If not
        specified, it stores it in terrahydro/toy_dataset

        Returns
        -------

        Path to the downloaded dataset

        """
        # URL to the evapotranspiration data on server
        url = 'https://portal.nccs.nasa.gov/datashare/astg/terrahydro/toy_dataset/'

        # Default path
        if not path:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'evapotranspiration')
        else:
            path = os.path.join(path, 'evapotranspiration')

        # Configuration for files to download
        files_config = [
            {'netcdf': 'FLUXNET.nc', 'zarr': 'FLUXNET.zarr'},
            {'netcdf': 'ERA5.nc', 'zarr': 'ERA5.zarr'}
        ]

        return DataServer._download_and_convert_zarr('evapotranspiration', url, path, files_config)
    
    @classmethod
    def load(cls, name, path=''):
        """ Downloads a registered data set by name """
        if name == 'mnist':
            return DataServer.load_mnist(path)
        
        if name == 'lstm_streamflow':
            return DataServer.load_streamflow(path)
        
        if name == 'soil_moisture':
            return DataServer.load_soil_moisture(path)
        
        if name == 'evapotranspiration':
            return DataServer.load_evapotranspiration(path)

        exc = f'No dataset called {name} on DataServer'
        raise ValueError(exc)
