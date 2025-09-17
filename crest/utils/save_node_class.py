import pickle as pkl
from pathlib import Path
import sys
import importlib
import inspect
import os
import logging
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

EXCLUDED_FIELDS = ['log', 'metadata', 'filepath', 'history', 'temp_cache']


@staticmethod
def import_module_from_path(module_path: str):
    """ Automatically imports the modules found in module_path 
        to allow automatic instatiation.
    """
    
    # check if module_path exists
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"Module file does not exist: {module_path}")

    # get spec from file_location
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(
        module_name, str(module_path))
    
    # use the the spec to get module
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # load module
    spec.loader.exec_module(module)

    # create dictionary of module names and their classes
    module_dict = {
        name: (module_name, module)
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module_name
    }

    return module_dict


class CustomUnpickler(pkl.Unpickler):
    """ Unpickles the object stored in "file" filepath."""

    def __init__(self, file, module_path=None):
        super().__init__(file)

        # Load all classes from the given module file, store mapping
        self._class_map = {}
        if module_path:
            self._class_map = import_module_from_path(module_path)

    def find_class(self, module, name):
        # if name is path, return path
        if name in ['WindowsPath', 'PosixPath']:
            return Path

        # if name in class map, return from dynamically loaded modules
        elif name in self._class_map:
            module_name, module = self._class_map[name]
            return getattr(module, name)

        # otherwise, import name from module
        return super().find_class(module, name)


class CustomPickler(pkl.Pickler):
    """ Pickles the object being passed with protocol. """

    def __init__(self, file, protocol=pkl.HIGHEST_PROTOCOL, exclude_fields=None):
        self.exclude_fields = exclude_fields or []
        super().__init__(file, protocol=protocol)

    def strip_fields(self, obj):
        """ Removes prespecified fields from the object. """

        for field in self.exclude_fields:
            if hasattr(obj, field):
                try:
                    setattr(obj, field, None)
                except Exception:
                    pass
        return obj

    def dump(self, obj):
        """ Saves pickle to drive. """

        obj = self.strip_fields(obj)
        super().dump(obj)


@staticmethod
def get_class_module_path(cls):
    """ Gets the file path of the module of passed object. """

    # get module based on module name
    module_name = cls.__module__
    module = sys.modules.get(module_name)

    # make sure we can get the file path of the module
    if module is None or not hasattr(module, '__file__'):
        raise ValueError(f"Cannot determine the path of module: {module_name}")

    # return file path
    return module.__file__


@staticmethod
def gen_filename(prefix='', suffix='.pkl', timestamp=True):
    """ Generate filename based on prefix, suffix and timestamp. """

    # get a hex uuid
    uid = uuid.uuid4().hex[:8]

    # generate timestamp if required
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if timestamp else ''

    # join prefix, suffix and timestamp together
    parts = [prefix, timestamp, uid]
    filename = "_".join(filter(None, parts)) + suffix

    # return filename
    return filename


@staticmethod
def read_pkl(filename, **kwargs):
    """ Load the pickled object using filename. """

    # get metadata_path if it exists 
    metadata_path = kwargs.get('metadata_path', 'outputs')
    metadata_file = os.path.join(str(metadata_path), 'metadata.json')

    # if it doesn't exist
    if (not os.path.exists(metadata_file)):
        logger.warning(f'metadata.json not found. {metadata_file} Falling back to raw inputs.')

        obj_file = filename
        module_path = kwargs.get('pathname', None)
    else: # if it exists
        logger.info(f"Found metadata at: {metadata_file}")

        # open it
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # set obj and module path parameters
        obj_file = metadata['obj_file']
        module_path = metadata['module_path']

    # if still not found, raise errors
    if (obj_file is None or not os.path.exists(obj_file)):
        raise FileNotFoundError(f"Object file not found: {obj_file}")

    if (module_path is None or not os.path.exists(module_path)):
        raise FileNotFoundError(f"Module file not found: {module_path}")

    # module path os required to successfully unpickle
    with Path(obj_file).open('rb') as f:
        unpickled = CustomUnpickler(f, module_path).load()

        # successful unpickling
        logger.info(f'Successfully unpickled object.')
        return unpickled


@staticmethod
def write_pkl(obj, filename, exclude_fields=None):
    """ Pickle the object to filename. """

    logger.info(
        f'write_pkl uses the root of the file path to also save obj metadata.')

    # file the root dir of the filename specified
    root_dir = Path(filename).parent
    module_path = get_class_module_path(obj.__class__)

    # create metadata.json file path based on root dir
    metadata_file = os.path.join(root_dir, 'metadata.json')

    # create dictionary of object information
    model_info = {
        "module_path": str(module_path),
        "obj_file": str(filename)
    }

    # save metadata into json formatted file 
    with open(metadata_file, 'w') as f:
        json.dump(model_info, f, indent=2)

        logger.info(f"Saved metadata to {metadata_file}")

    # pickle the object and save it to the file path specified
    with Path(filename).open('wb') as f:
        CustomPickler(
            f, exclude_fields=exclude_fields or EXCLUDED_FIELDS).dump(obj)
        logger.info(f"Saved pickled object to {filename}")
