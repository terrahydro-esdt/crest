import logging

from crest.data.loading.StructuredDataset import StructuredDataset
from urllib.request import urlretrieve
import os
import pickle as pkl
import xarray as xr

logger = logging.getLogger(__name__)


class DataServer:
    """ Data Server for loading files from crest server"""

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

        # Make directory if it doesn't exist
        if not os.path.isdir(path):
            os.mkdir(path)

        # list of the required files
        names = ['SMAP.nc',
                 'ERA5.nc',
                 'Soil.nc',
                 'Irrigation.nc']
        zarr_names = ['SMAP.zarr',
                      'ERA5.zarr',
                      'StaticAttributes.zarr/Soil',
                      'StaticAttributes.zarr/Irrigation']
        for index, n in enumerate(names):
            # Download files which are in
            # netcdf formats
            file = os.path.join(path, n)
            file_zarr = os.path.join(path, zarr_names[index])
            if not os.path.isfile(file) and not os.path.exists(file_zarr):
                try:
                    urlretrieve(url + n, file)
                except Exception as e:
                    print(e)
                    return None
            
                # convert files to the zarr format
                data = xr.open_dataset(file)
                if n in ['SMAP.nc', 'ERA5.nc']:
                    data.to_zarr(os.path.join(path, n.split('.')[0]+'.zarr'))
                else:
                    data.to_zarr(os.path.join(path, 'StaticAttributes.zarr/'+n.split('.')[0]))
                
                # remove the netcdf files
                os.remove(file)
        
        print('Download complete at', path)
        return path
            
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
        # URL to the soil_moisture data on server
        url = 'https://portal.nccs.nasa.gov/datashare/astg/terrahydro/toy_dataset/'

        # Default path
        if not path:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'evapotranspiration')
        else:
            path = os.path.join(path, 'evapotranspiration')

        # Make directory if it doesn't exist
        if not os.path.isdir(path):
            os.mkdir(path)

        # list of the required files
        names = ['FLUXNET.nc',
                 'ERA5.nc',]
        zarr_names = ['FLUXNET.zarr',
                      'ERA5.zarr',]
        for index, n in enumerate(names):
            # Download files which are in
            # netcdf formats
            file = os.path.join(path, n)
            file_zarr = os.path.join(path, zarr_names[index])
            if not os.path.isfile(file) and not os.path.exists(file_zarr):
                try:
                    urlretrieve(url + n, file)
                except Exception as e:
                    print(e)
                    return None
            
                # convert files to the zarr format
                data = xr.open_dataset(file)
                data.to_zarr(os.path.join(path, n.split('.')[0]+'.zarr'))
                
                # remove the netcdf files
                os.remove(file)
        
        print('Download complete at', path)
        return path
    
    @classmethod
    def load(cls, name, path=''):
        if name == 'mnist':
            return DataServer.load_mnist(path)
        
        if name == 'lstm_streamflow':
            return DataServer.load_streamflow(path)
        
        if name == 'soil_moisture':
            return DataServer.load_soil_moisture(path)
        
        if name == 'evapotranspiration':
            return DataServer.load_evapotranspiration(path)

        exc = f'No dataset called {name} on DataServer'
        raise Exception(exc)
