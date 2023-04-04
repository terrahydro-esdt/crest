from functools import wraps
from itertools import starmap
from typing import get_args, get_origin, _type_repr
from typing import Union, Iterator, TypeVar #Generic, _SpecialGenericAlias, _GenericAlias
from types import UnionType #GenericAlias
from abc import ABCMeta, ABC

import inspect, copy


def ensure_types(cls, f):
    """ Ensure type annotations are followed, raising TypeError if not """

    def type_repr(val=None, T=None):
        """ Recursive type representation """
        if isinstance(val, Iterator): val = copy.deepcopy(val)
        if isinstance(T, str):        return T
        if T is not None:             return _type_repr(T)
        container = _type_repr(type(val)).split('.')[-1]

        # Recurse on T if it's iterable (and not a str)
        if hasattr(val, '__iter__') and not isinstance(val, str):
            ele_types = ', '.join(set(map(type_repr, val)))
            container = f'{container}[{ele_types}]'
        return container

    def equal_tuples(val, T):
        """ Check that val and T are tuples of the same length """
        is_tuple = isinstance(val, tuple) and isinstance(T, tuple)
        return is_tuple and (len(val) == len(T))

    def handle_generic(val, T):
        """ Swap generic TypeVar for val type """
        if isinstance(T, TypeVar) and len(T.__constraints__):
            raise NotImplementedError('TypeVar constraints not implemented')
        if isinstance(T, TypeVar): T = T.__bound__ or type(val)
        elif equal_tuples(val, T): T = tuple(map(handle_generic, val, T))
        elif T is None:            T = type(None)
        return T

    def istype(val, T):
        """ Recursively determine if value matches generic type T """
        # Ensure Iterators aren't modified, and handle TypeVars
        if isinstance(val, Iterator): val = copy.deepcopy(val)
        T = handle_generic(val, T)

        # Handle a tuple of types
        if equal_tuples(val, T):   return all(map(istype, val, T))
        elif isinstance(T, tuple): return False

        # Try a simple type check, which fails if T is a parameterized generic
        try:              return isinstance(val, T)
        except TypeError: pass

        # Get origin type and parameterized types 
        origin = get_origin(T)
        types  = get_args(T)

        # Origin is just a union of types, so we can check for any valid
        if origin in [Union, UnionType]: 
            return any(istype(val, T) for T in types)

        # String with class name might be used in the class definition
        if (origin is None) and isinstance(T, str):
            cls_eq = lambda cls: getattr(cls, '__name__', '') == T
            return any(map(cls_eq, (type(val),) + val.__class__.__bases__))

        # Ensure it matches the origin type
        if isinstance(val, origin):

            # If val can't be iterated, i.e. is not a container type like list
            if not hasattr(val, '__iter__'): return True

            # Recursively type check the container elements, replacing
            # T with the type of the first element when necessary
            elems = getattr(val, 'items', lambda: [[v] for v in val])()
            first = (list(elems) + [[]])[0]
            types = list(map(handle_generic, first, types))
            return all(all(map(istype, v, types)) for v in elems)

    @wraps(f)
    def wrapper(*args, **kwargs):
        """ Wrap the function with an explicit type checker """
        if not hasattr(f, '__code__'): return f(*args, **kwargs)

        # Extract the function parameters and respective annotations
        f_name   = f.__code__.co_name
        f_params = f.__code__.co_varnames 
        f_types  = f.__annotations__
        keywords = inspect.getcallargs(f, *args, **kwargs)

        def check_type(key, value):
            if key in f_types and not istype(value, f_types[key]):
                require = type_repr(T=f_types[key])
                actual  = type_repr(value)
                message = f'{cls}.{f_name} parameter "{key}" must be'
                message+= f' of type {require}, but found type {actual}'
                raise TypeError(message)
            return value

        # Iterate over all of the function parameters and check types
        list(starmap(check_type, keywords.items()))
        return check_type('return', f(*args, **kwargs))
    return wrapper



class BaseMeta(ABCMeta):
    """ Metaclass for class name string representation. """
    def __str__(self):  return f"<class '{self.__name__}'>"
    def __repr__(self): return f"<class '{self.__name__}'>"

    def __new__(cls, *args, **kwargs):
        """ Wrap __init__ with type checking """
        cls = type.__new__(cls, *args, **kwargs)
        cls.__init__ = ensure_types(cls, cls.__init__)
        return cls




class BaseAbstract(ABC, metaclass=BaseMeta):
    """ Base class for any other 'Base' classes. """
    def __str__(self):  return self.__class__.__name__
    def __repr__(self): return self.__class__.__name__


    def __getattribute__(self, attr):
        """ Provides type checking for class functions that use annotations """
        f = object.__getattribute__(self, attr)
        if attr[:2] == '__': return f 

        # If this is a function, check types for the given inputs
        return ensure_types(self, f) if callable(f) else f
    

    @classmethod
    def load(cls, obj, *args, **kwargs):
        """ Wrap an object with the parent class if it isn't already one """
        if not isinstance(obj, cls):
            obj = cls(obj, *args, **kwargs)
        return obj