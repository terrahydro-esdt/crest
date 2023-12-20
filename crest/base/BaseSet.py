from collections.abc import Collection, Callable, Iterator
from itertools import zip_longest, starmap, compress
from operator import itemgetter
from typing import TypeVar, Any

import numpy as np
import dask 
import operator 

from .BaseAbstract import BaseAbstract


# Generic representing single type
T = TypeVar('T')


class BaseSet(BaseAbstract):
    """Class which wraps a set of classes into a single object.

    Notes
    -----
    This class allows operations over sets of classes. For example,
    a Blockset(BaseSet) class would hold a set of Block objects, and
    allow functions to be applied seamlessly to all Block objects within
    its container (e.g. Blockset.resolution would return a list of 
    Block.resolution values, assuming 'resolution' is a property of Block).

    In addition, a list of parameters can be mapped over the set elements
    by calling the same function and passing _map=[parameters] as a keyword.
    Similarly, keywords can be used via _kwmap={'param': parameters}. 

    Parameters
    ----------
    objs : Collection
        Collection of objects that this class is wrapping. This needs to be
        a homogeneous collection, with all objects the same class. 

    Examples
    --------
    >>> class StringSet(BaseSet): pass
    >>> strings = StringSet(['a b c', 'd.e.f'])
    >>> strings.upper()
    StringSet['A B C', 'D.E.F']
    >>> strings.join(['1 ', ' 2'])
    StringSet['1 a b c 2', '1 d.e.f 2']
    >>> strings.split(_map=[[' ', '.']])
    StringSet[['a', 'b', 'c'], ['d', 'e', 'f']]
    >>> strings.split(_map=[[' ', '.'], [1, 2]])
    StringSet[['a', 'b c'], ['d', 'e', 'f']]
    >>> strings.split(_map=[[' ', '.']], _kwmap={'maxsplit': [1, 2]})
    StringSet[['a', 'b c'], ['d', 'e', 'f']]
    >>> strings.upper().split(' ')
    StringSet[['A', 'B', 'C'], ['D.E.F']]

    """ 
    def __init__(self, objs: Collection[T]):
        self.container = objs


    def __str__(self) -> str:
        return repr(self)


    def __repr__(self) -> str:
        """ String representation: BaseSet[container] """
        return f"{super().__repr__()}{getattr(self, 'container', '')}"


    def __eq__(self, other: Any) -> bool:
        """ Check for equality with another container """
        if isinstance(other, BaseSet) and len(self) == len(other):
            return all(a == b for a,b in zip(self, other))
        return self._wrap([a == other for a in self]) 


    def __len__(self) -> int:
        """ Number of elements in this set """
        return len(self.container)


    def __iter__(self) -> Iterator[T]:
        """ Iterate over the elements in this set """
        yield from self.container


    def __contains__(self, obj: T) -> bool:
        """ Check if obj is in this container """
        return obj in self.container


    def __getitem__(self, idx: Any) -> T:
        """ Get an element in the container """
        # Wrap the sliced container in a new BaseSet 
        if isinstance(idx, slice):
            return self.__class__(self.container[idx])

        # Allow selecting via a collection of ints or bools
        if hasattr(idx, '__len__'):
            if all(isinstance(i,int) and not isinstance(i,bool) for i in idx):
                return self.__class__([self[i] for i in idx])
            return self.__class__(list(compress(self, idx)))
        return self.container[idx]
    

    def __setitem__(self, idx: Any, val: T) -> None:
        """ Set an element in the container """
        self.container[idx] = val


    def __array__(self, *args, **kwargs):
        return np.array(self.container)


    def _repr_html_(self) -> str:
        """ Format the set nicely for notebooks """
        item_rep = lambda item: getattr(item, '_repr_html_', item.__repr__)()
        sub_html = ''.join(map(item_rep, self.container))
        return f'<h3>{self}:</h3><div>{sub_html}</div>' 


    def _wrap(self, objs: Collection) -> 'BaseSet':
        """ Wrap the return objects in either the original 
            *Set class if the obj type hasn't changed, or
            in a BaseSet class otherwise """ 
        if isinstance(objs[0], self.container[0].__class__):
            return self.__class__(objs)
        return BaseSet(objs)


    def __getattr__(self, attr: str) -> 'BaseSet':
        """ Allow calls to be passed to the objects composing this set """
        if attr == 'container': return object.__getattribute__(self, attr)

        # Ignore pickle functions to avoid recursion issues
        if attr in ['__getstate__', '__setstate__', '__array_struct__', '__array_interface__']:
            raise AttributeError()

        objs_attr = [getattr(obj, attr) for obj in self]
        if not callable(objs_attr[0]): 
            return self._wrap(objs_attr)
        
        def wrapper(*args, _map=[], _kwmap={}, _delay=True, __objs=objs_attr, **kwargs) -> 'BaseSet':
            """ Wrapper function which allows distributing parameters 
                over the container objects. 

            Parameters
            ----------
            *args
                All unnamed arguments are passed to all container objects.
            _map   : list[list]
                Any parameters passed to _map are distributed over container
                objects - and so the length of each _map element must equal
                the number of objects in the container. 
            _kwmap : dict[str, list]
                Equivalent to _map, but allows using keyword arguments for
                the distributed parameters. 
            **kwargs
                Equivalent to *args, but allows using keyword arguments to
                pass the same value to all container objects.

            Returns
            -------
            BaseSet 
                Returns a *Set object whose container is a list of the
                results from calling the function referred to by `attr`
                on all of the current *Set container's objects. 

            Notes
            -----
            - Dask is used to perform the operation over the container
            objects in parallel
            - See the BaseSet docstring for examples

            """
            # Ensure any generators are replaced
            _map   = [list(m) for m in _map]
            _kwmap = {k: list(v) for k, v in _kwmap.items()}

            # Check that all passed arguments are the correct length
            valid = lambda v: len(v) in [0, len(__objs)]
            items = _map + list(_kwmap.values())
            assert(all(map(valid, items))), \
                f'{self} has {len(__objs)} items, but items passed via' + \
                f' _map/_kwmap had sizes {map(len, items)}: {_map} | {_kwmap}'

            create_dict = lambda v: dict(zip(_kwmap.keys(), v))
            get_outputs = lambda f, k, *a: f(*(args+a), **(kwargs|k))

            # Use a delayed call over the functions to compute simultaneously
            if _delay: __objs = map(dask.delayed, __objs)

            # Distribute arguments over the container objects
            kwdict = map(create_dict, zip(*_kwmap.values()))
            params = map(tuple, [kwdict] + _map)
            f_args = zip_longest(__objs, *params, fillvalue={})
            output = starmap(get_outputs, f_args)

            # Only exhaust the generator if _compute was requested
            if _delay: output = dask.compute(*output)
            return self._wrap(list(output))
        return wrapper


    def sort(self, 
        function  : Callable   | None = None,
        container : Collection | None = None,
    ) -> Collection:
        """Sort the container with the given function. 
        
        Parameters
        ----------
        function  : Callable   | None
            Sorting function to use. This function should take one input
            parameter, which is the current element to sort; and return 
            one value, which is the value to use for that element when sorting.
            If no function is given or function is None, the container is 
            sorted according to the original ordering of the data (self.order).
        container : Collection | None
            The container to sort. If no container is given, the container
            of this BaseSet (self.container) is used.

        Returns
        -------
        Collection
            The collection which was sorted. If no collection was given as 
            input, this will be self.container; otherwise it is the sorted
            container which was given as input. 

        """
        
        # If we're reordering, first return to the original order
        if function is not None:
            self.sort(container=container)

        overwrite = container is None
        container = container if not overwrite else self.container 

        # If no function is given, use the existing order to sort
        if function is None:
            if getattr(self, 'order', None) is None:
                return container

            assert(len(container) == len(self.order))
            target = zip(self.order, container)
            sorter = lambda i: i[0]
        
        else:
            target = enumerate(container)
            sorter = lambda i: function(i[1])

        # Apply the requested sorting and get the new container and order
        new_order, container = map(list, zip(*sorted(target, key=sorter)))

        # Overwrite the old container and order if no container was given
        if overwrite:
            self.order     = new_order
            self.container = container
        return container


    @property
    def ix(self):
        """ Convenience function to slice container's objects """
        def getitem(_, i):
            multi = lambda x: isinstance(x, Collection)
            if not multi(i): i = [i] * len(self)
            assert(len(i) == len(self))
            
            values = []
            for j, c in zip(i, self):
                val = itemgetter(*np.atleast_1d(j))(c)
                if multi(j) and not multi(val):
                    val = [val]
                values.append(val)
            return self._wrap(values)
        return type('ix', (object,), {'__getitem__': getitem})()


    def map(self, f: Callable):
        """ Convenience function to map a callable over container's objects """
        return self._wrap(list(map(f, self)))


    def __new__(cls, *args, **kwargs):
        """ Add special operators to the BaseSet (e.g. __add__) """
        def factory(name):
            """ Allow operators to be distributed over the container """
            def wrapper(self, other=None):
                method = lambda a,b: getattr(a, name)(b)
                if other is not None:
                    if isinstance(other, Collection) and (len(other) == len(self)):
                        result = [method(c, o) for c,o in zip(self, other)]
                    else: result = [method(c, other) for c in self]
                else: result = [getattr(c, name)() for c in self]
                assert(not any([r is NotImplemented for r in result]))
                return self._wrap(result)
            wrapper.__name__ = name 
            return wrapper

        # Add all Operators except those already contained in the class
        skip = ['__name__', '__loader__', '__package__'] + dir(cls)
        for op in dir(operator):
            if op.startswith('__') and op.endswith('__') and (op not in skip):
                try:              setattr(cls, op, factory(op))
                except TypeError: print(f'Failed to set {op}: {e}')
        return super().__new__(cls, *args, **kwargs)