from crest.base import BaseAbstract
from collections.abc import Collection
from collections import UserDict
import numpy as np
import tensorflow as tf
from .TensorSpec import TensorSpec 
import dill

class IOSpec(UserDict):
    """
   IOSpec is a managed dictionary that defines 
   a collection of specs for a set of keys. It
   can generate the inputs/outputs dictionary of keys : TensorSpecs()
   using IOSpec.spec, and can be used to create data sets using
   crest.Dataset.from_specs(). Elements in IOSpec
   are of the form:

   {str(label) : {'keys' : list[str],
                  'labeled' : bool,
                  'coord_shapes': dict[str,dict[str, Collection[int] | int | None]],
                  'dtype' : 'float32'
                }
            }

    'labeled' = whether to add label to the keys
    'coord_shapes' = specification of coords and window extents. If only
     one coord_shape is given, it is applied to all keys.

     IOSpec can be initialized with a dictionary of specs or elements can
     be added as:  IOSpec[label] = {'keys' : list[str],
                  'labeled' : bool,
                  'coord_shapes': dict[str,dict[str, Collection[int] | int | None]],
                  'dtype' : str | dict
                }

    Parameters
    ----------

    io : optional, dictionary of IOSpec elements following the above format.
    
    """

    def __init__(self,io = None):
        super().__init__()
        if io:
            for k,v in io.items():
                self[k] = v

    def __setitem__(self, label: str, value: dict):
        if not isinstance(value,dict):
            raise ValueError("IOSpec items must be dictionaries")
        
        io_spec = {
                   'keys' : None,
                   'labeled' : True,
                   'coord_shapes': None,
                   'dtype' : 'float32'
                    }
        
        # Check if correct dictionary
        for i in ['keys','coord_shapes']:
            if not i in value:
                raise ValueError(f'Elements must define {i}')

        # Check if unallowed keys are given
        unallowed = [k for k in value if k not in io_spec]
        if unallowed:
            raise ValueError(f'IOspec keys = {unallowed} not allowed')
         
        io_spec.update(value)

        if not isinstance(io_spec['keys'],list):
            io_spec['keys']  = [io_spec['keys']]

        # Apply to all if only one specified
        is_coord_dict = [] 
        is_coord_dict = [k for k in io_spec['keys'] if k in io_spec['coord_shapes']]
        if not is_coord_dict:
            io_spec['coord_shapes'] = {k:io_spec['coord_shapes'] for k in io_spec['keys']}
        else:
            if not list(io_spec['coord_shapes'].keys()) == io_spec['keys']:
                raise ValueError('coord_shapes dictionary does not have the right keys')

        # Apply shape to all    
        if not isinstance(io_spec['dtype'],dict):
           io_spec['dtype'] = {k : io_spec['dtype'] for k in io_spec['keys']}
        

        super().__setitem__(label,io_spec)


    def _genshape(self,coord_shape) -> list[int | None]:
        """ generate the shape of the tensors from the coord_shape """

        # If coord specified for each feature
        if not len(coord_shape): return [None]
        # Handle left/right extent being used, and order by coordinate key
        get_total = lambda s: sum(s)+1 if isinstance(s, Collection) else s
        _,ordered = zip(*sorted(coord_shape.items(), key=lambda kv:kv[0]))
        return [None] + list(map(get_total, ordered))
        
    @property
    def spec(self) -> dict:
        """ Generate io spec dictionary specifying {key : TensorSpec,...}"""
        spec = {}
        
        for k,v in self.items():
            # Add labels if appropriate
            for key in v['keys']:
                shape = self._genshape(v['coord_shapes'][key])

                if v['labeled']:
                    _key = k + '>>' + key

                ts = {'shape' : shape, 'name' : _key, 'dtype' : v['dtype'][key]}
                spec[_key] = TensorSpec(ts)

        return spec
    
    @property
    def keys(self) -> list:
        """ Generates a list of all labeled keys """
        return list(self.spec.keys())
    
    @property
    def labels(self) -> list:
        """ Generates a list of all labels """
        return list(self.keys())
    
    @property
    def coord_shapes(self) -> dict:
        """ Generates a dict of all coord_shapes """
        coord_shapes = {}
        
        for k,v in self.items():
            # Add labels if appropriate
            for key in v['keys']:
                if v['labeled']:
                    _key = k + '>>' + key
                coord_shapes[_key] = v['coord_shapes'][key]
        return coord_shapes
    
    def __repr__(self):
        return f'IOSpec({str(list(self.keys()))})'
    
    def encode(self):
        encode = dill.dumps(self.__dict__) 
        return encode
    
    @classmethod
    def decode(cls,encode):
        decode = dill.loads(encode)
        io_spec = IOSpec()
        io_spec.__dict__.update(decode)
        return io_spec