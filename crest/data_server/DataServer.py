from crest.data.loading.StructuredDataset import StructuredDataset
from urllib.request import urlretrieve
import os
import pickle as pkl
import xarray as xr

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

    @classmethod
    def load(cls, name, path=''):
        if name == 'mnist':
            return DataServer.load_mnist(path)
        
        if name == 'lstm_streamflow':
            return DataServer.load_streamflow(path)

        exc = f'No dataset called {name} on DataServer'
        raise Exception(exc)
