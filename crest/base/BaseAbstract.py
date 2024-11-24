from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import get_args, get_origin, _type_repr
from typing import Union, Iterator, TypeVar
from types import UnionType
from abc import ABC

import weakref
import inspect


def type_repr(val=None, T=None, _maxdepth: int = 4):
    """ Recursive version of typing._type_repr for type representation """
    if isinstance(T, str):        return T
    if T is not None:             return _type_repr(T)
    if isinstance(val, Iterator): return type(val)
    container = _type_repr(type(val)).split('.')[-1]

    # Recurse on val if it's iterable (and not a str)
    if hasattr(val, '__iter__') and not isinstance(val, str) and _maxdepth:
        recurse = partial(type_repr, T=T, _maxdepth=_maxdepth - 1)
        try:
            item = next(iter(getattr(val, 'items', lambda: val)()))
            if hasattr(val, 'items'):
                ele_type = ', '.join(map(recurse, item))
            else:
                ele_type = recurse(item)
        except:
            ele_type = '?'
        return f'{container}[{ele_type}]'
    return container


def equal_tuples(val, T):
    """ Check that val and T are tuples of the same length """
    is_tuple = isinstance(val, tuple) and isinstance(T, tuple)
    return is_tuple and (len(val) == len(T))


def handle_generic(val, T):
    """ Swap generic TypeVar for val type """
    if isinstance(T, TypeVar) and len(T.__constraints__):
        raise NotImplementedError('TypeVar constraints not implemented')
    if isinstance(T, TypeVar):
        T = T.__bound__ or type(val)
    elif equal_tuples(val, T):
        T = tuple(map(handle_generic, val, T))
    elif T is None:
        T = type(None)
    return T


def istype(val, T):
    """ Recursively determine if value matches generic type T """
    # Cannot look inside iterators to verify types, as it would exhaust values
    if isinstance(val, Iterator):
        origin = get_origin(T) or T
        types = get_args(T)
        if origin in [Union, UnionType]:
            return any(istype(val, T) for T in types)
        return isinstance(val, origin)

    T = handle_generic(val, T)

    # Handle a tuple of types
    if equal_tuples(val, T):
        return all(map(istype, val, T))
    elif isinstance(T, tuple):
        return False

    # Try a simple type check, which fails if T is a parameterized generic
    try:
        return isinstance(val, T)
    except TypeError:
        pass

    # Get origin type and parameterized types 
    origin = get_origin(T)
    types = get_args(T)

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



class EnsureTypes:
    """Ensure type annotations are followed, raising TypeError if not.

    Wrapping with a class rather than a function allows access to the 
    underlying object attributes when a callable object is wrapped.

    """

    def __init__(self, cls_obj: 'BaseAbstract', callable_obj: Callable):
        self._cls_repr = repr(cls_obj)
        self._callable = callable_obj


    def __repr__(self):
        return f'{self._cls_repr}.{self._callable.__code__.co_name}'


    def __call__(self, *args, **kwargs):
        """ Wrap the function with an explicit type checker """

        # Extract the function parameters and respective annotations
        function = self._callable
        annotate = function.__annotations__
        keyvalue = inspect.getcallargs(function, *args, **kwargs)
        keyvalue |= {'return': function(*args, **kwargs)}

        # Iterate over all parameters and verify types match the annotations
        [self.verify_type(value, annotate[key], f'{self} parameter "{key}"')
         for key, value in keyvalue.items() if key in annotate]
        return keyvalue['return']


    def __getattr__(self, attr):
        """ Pass through attribute lookups to the underlying callable """
        return self if attr == '__call__' else getattr(object.__getattribute__(self, '_callable'), attr)


    @classmethod
    def wrap(cls, obj, obj_attr):
        """ Wrap the object attribute if valid, and return it otherwise """
        # 1) not EnsureTypes; 2) callable; 3) not bytecode; 4) annotated
        if (not isinstance(obj_attr, cls)
                and callable(obj_attr)
                and hasattr(obj_attr, '__code__')
                and getattr(obj_attr, '__annotations__', {})):
            return cls(obj, obj_attr)
        return obj_attr


    @classmethod
    def verify_type(cls, obj, annotation, label):
        """ Raise TypeError if obj type does not match the given annotation """
        try:
            invalid_type = not istype(obj, annotation)
        except TypeError:
            raise TypeError(f'{label} annotation "{annotation}" is not valid')

        if invalid_type:
            req = type_repr(T=annotation)
            typ = type_repr(obj)
            msg = f'{label} must be of type {req}, but found type {typ}'
            raise TypeError(msg)


    @property
    def object(self):
        """ Try to return the underlying object this function is bound to """
        return getattr(self._callable, '__self__')



class BaseAbstract(ABC):
    """ Base class for any other 'Base' classes. """

    def __repr__(self):
        return self.__class__.__name__


    def __getattribute__(self, name):
        """ Provides type checking for class functions that use annotations """
        attr = object.__getattribute__(self, name)
        return attr if name.startswith('__') else EnsureTypes.wrap(self, attr)


    def __new__(cls, *args, **kwargs):
        """ Called whenever a new inheriting class object is instantiated """
        # Store a weakref of the object to allow tracking object persistance
        obj = super().__new__(cls)
        cls._refs[id(obj)] = obj
        return obj


    def __init_subclass__(cls, *args, **kwargs):
        """ Called when an inheriting class is defined.
            Wraps __init__ with type checking, and allows 
            __post_init__ functions in inheriting classes.
        """

        def init_decorator(init):
            def __init__(self, *args, **kwargs):
                init(self, *args, **kwargs)
                # Only call for the final __init__ in the inheritance stack
                if type(self) is cls: self.__post_init__()

            return __init__

        type_checked = EnsureTypes.wrap(cls, cls.__init__)
        cls.__init__ = init_decorator(type_checked)
        cls._refs = weakref.WeakValueDictionary()


    def __post_init__(self):
        """ Allows inheriting classes to define a function that runs after
            the __init__ method; mainly useful for Base classes to force
            children to perform some operations after initialization """
        pass


    @classmethod
    def load(cls, obj, *args, **kwargs):
        """ Wrap an object with the parent class if it isn't already one """
        return obj if isinstance(obj, cls) else cls(obj, *args, **kwargs)


    @classmethod
    def interactive(cls, environment: dict = {}, n_prior_frames: int = 0):
        """ Start an interactive console wherever this function is called.

        Parameters
        ----------
        environment : dict
            Any extra objects to include in the console environment.
        n_prior_frames : int
            Extra frames to rewind when getting the calling context for this
            function. If `interactive` is called directly, this value should
            be 0 to use the context where it was called (default); if this
            function is e.g. called by a helper function and so the desired
            context is where the helper was used at, the helper should use
            `interactive(n_prior_frames=1)`; etc.

        """

        # Need to step one additional frame backwards if this function was
        # called from an instance of a class, rather than by a class itself,
        # due to type check wrapping performed in this file
        filename = inspect.currentframe().f_back.f_code.co_filename
        n_prior_frames += Path(filename).stem == 'BaseAbstract'

        from crest.utils import interactive
        interactive(environment, n_prior_frames + 1)


    @classmethod
    def verify_type(cls, obj, annotation, label=''):
        """ Allow a class to verify arbitrarily complex types manually """
        EnsureTypes.verify_type(obj, annotation, label or f'{cls}: {obj}')
