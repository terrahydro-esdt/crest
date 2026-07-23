""" 
This module implements the HTG registry. 
The registry tracks all subclasses that inherit
from HTG to allow saving and loading with dynamic
imports.
"""

import importlib

class Registry:
    """ 
   
    class registry for HTG 

    TODO: Registry does not handle
    classes with the same name. It will throw an error.
    
    """
    def __init__(self):
        self.registry = {}
        self.registry['HierarchalTensorGraph'] = {
            'name' : 'HierarchalTensorGraph',
            'module' : 'crest.model.HierarchalTensorGraph'
        }

    def register(self,cls):
        """ registered extension of HTG """
        name = cls.registry_name

        d = {
            'name' : cls.__name__,
            'module' : cls.__module__
        }

        if name in self.registry.keys():
            if self.registry[name] != d:
                raise Exception(f'HTG {name} already exist in the registry.')

        self.registry[name] = d

    def __getitem__(self,name):
        """ Returns the type associated with the registry name """
        mod = importlib.import_module(self.registry[name]['module'])
        obj = getattr(mod, self.registry[name]['name'])
        return obj
