from collections.abc import Mapping

import warnings
import reprlib
import json


def json_safe(value, as_string: bool = False, name: str = 'attrs'):
    """ Recursively convert object to safe types for json serialization """
    warn = lambda s: warnings.warn(s, UserWarning, stacklevel=4)
    
    def parse(obj, path: str):
        """ Recursively convert object and warn user of invalid types """
        try:              return json.loads(json.dumps(obj))
        except TypeError: pass
            
        if isinstance(obj, Mapping):
            # First verify all dict keys are strings
            for key in obj:
                if not isinstance(key, str):
                    val = reprlib.repr(key)
                    warn(f'Casting key {path}.{val} ({type(key)=}) to str')
            return {str(k): parse(v, f'{path}.{k}') for k,v in obj.items()}

        if isinstance(obj, set):
            val = reprlib.repr(obj)
            warn(f'Casting object {path}={val} ({type(obj)=}) to list')
            obj = list(obj)
            
        if isinstance(obj, (list, tuple)):
            return [parse(o, f'{path}[{i}]') for i,o in enumerate(obj)]

        val = reprlib.repr(obj)
        warn(f'Casting object {path}={val} ({type(obj)=}) to str')
        return str(obj)
    
    safe = parse(value, name)
    if as_string:
        safe = json.dumps(safe)
    return safe