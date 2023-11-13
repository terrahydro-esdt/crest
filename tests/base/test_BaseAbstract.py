from itertools import product
from typing import _type_repr, Dict, List, Union, Optional, TypeVar, Iterator
from copy import deepcopy
from abc import abstractmethod

import pytest
import re

import sys
print(sys.path)

import crest
import pkgutil

modualList = []
for importer, modname, ispkg in pkgutil.iter_modules(crest.__path__):
    modualList.append(modname)

print(modualList)

from crest.crest.base.BaseAbstract import BaseAbstract

def type_repr(val=None, T=None):
    """ Recursive type representation """
    if isinstance(val, Iterator): val = deepcopy(val)
    if isinstance(T, str):        return T
    if T is not None:             return _type_repr(T)
    container = _type_repr(type(val)).split('.')[-1]

    # Recurse on T if it's iterable (and not a str)
    if hasattr(val, '__iter__') and not isinstance(val, str):
        try:    ele_types = type_repr(next(iter(val)))
        except: ele_types = '?'
        container = f'{container}[{ele_types}]'
    return container


def error(param, expected, found):
    """ Create the error that should be raised """
    message = f'parameter "{param}" '
    message+= f'must be of type {type_repr(T=expected)},'
    message+= f' but found type {type_repr(found)}'
    return pytest.raises(TypeError, match=re.escape(message))


def run(T, valid, invalid):
    """ Dynamically create a Test class with the given annotations """
    T_a = T.get('a', None)
    T_b = T.get('b', None)
    T_c = T.get('c', None)
    T_r = T['return'] = T.get('return', None), 'Test'

    class Test(BaseAbstract):
        def __init__(self, a:T_a=None, b:T_b=None, c:T_c=None, **kwargs): pass
        def function(self, a:T_a=None, b:T_b=None, c:T_c=None, **kwargs)-> T_r:
            return kwargs.get('return', None), self

    # Test all valid combinations
    keys, values = zip(*valid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    valid = combinations[0]
    inst  = [Test(**c).function(**c)[1] for c in combinations][0]

    # Test all invalid options
    for param, options in invalid.items():
        for option in options:
            for f in [Test, inst.function]:
                test_kws = dict(valid)
                test_kws[param] = option

                # Skip the invalid init and format the return value
                if param == 'return':
                    if f is Test: continue
                    option = (option, inst)

                with error(param, T[param], option): 
                    f(**test_kws)
                    print(f'param={param}  option={option}')
                    print(f'f={f}  test_kws={test_kws}')

    

def test_single():
    """ Test single type """
    run(T = {
            'a' : int, 
            'b' : str, 
            'c' : float,
        },
        valid = {
            'a' : [1],
            'b' : ['k'],
            'c' : [1.],
        },
        invalid = {
            'a' : ['a', 1.],
            'b' : [1, None],
            'c' : ['a', 1],
        },
    )


def test_list():
    """ Test list annotation """
    run(T = {
            'a' : list,
            'b' : list[str],
            'c' : List[int],
        },
        valid = {
            'a' : [[], [1]],
            'b' : [[], ['k']],
            'c' : [[], [1]],
        },
        invalid = {
            'a' : [1, {}],
            'b' : [1, [1], ['k', 1]],
            'c' : [1, ['k'], ['k', 1]],
        },
    )


def test_dict():
    """ Test dictionary annotation """
    run(T = {
            'a' : dict,
            'b' : dict[str, int],
            'c' : Dict[str, str],
        },
        valid = {
            'a' : [{}, {'k':'v'}],
            'b' : [{}, {'k': 1, 'j': 2}],
            'c' : [{}, {'k': 'v'}],
        },
        invalid = {
            'a' : [1, None],
            'b' : [1, {'k':'v'}, {'k': 'a', 'j': 2}],
            'c' : [1, ['k'], {1: 'a'}],
        },
    )


def test_nested():
    """ Test nested list/dict annotation """
    run(T = {
            'a' : dict[tuple, list[str]],
            'b' : list[dict[str, str]],
            'c' : List[list[Dict[int, str]]],
        },
        valid = {
            'a' : [{}, {(1,):[]}, {(1,2): ['a'], (3,4): ['b']}],
            'b' : [[], [{},{}], [{'a':'b'}, {'c':'d', 'e':'f'}]],
            'c' : [[], [[{},{}], []], [[{1:'a'}], [{2:'b', 3:'c'}, {4:'d'}]]],
        },
        invalid = {
            'a' : [1, {'a':[]}, {(1,):[2]}],
            'b' : [1, [{'a':1}], [{'k': 'a'}, {'j': 'b', 'i':2}]],
            'c' : [1, [[{'k': 'a'}]], [{1: 'a'}], [[{1:'a'}, {'k': 'b'}]]],
        },
    )


def test_union():
    """ Test union annotation """
    run(T = {
            'a' : int | str, 
            'b' : str | list[int] | int, 
            'c' : Union[int, float],
        },
        valid = {
            'a' : [1, 'a'],
            'b' : ['k', [], [1], 1],
            'c' : [1, 1.],
        },
        invalid = {
            'a' : [1., [1]],
            'b' : [1., ['a']],
            'c' : ['a'],
        },
    )


def test_optional():
    """ Test optional annotation """
    run(T = {
            'a' : int | None, 
            'b' : Optional[float],
            'c' : list[Optional[str]], 
        },
        valid = {
            'a' : [1, None],
            'b' : [1., None],
            'c' : [[], ['a'], [None]],
        },
        invalid = {
            'a' : [1., [1]],
            'b' : [1, ['a']],
            'c' : [[1], [[None]]],
        },
    )


def test_generic():
    """ Test generic types """
    T = TypeVar('T')
    B = TypeVar('B', bound=int | str)
    run(T = {
            'a' : T, 
            'b' : list[T],
            'c' : dict[B, T], 
        },
        valid = {
            'a' : [1, None, 2., 'a', [1, 'a']],
            'b' : [[], [1, 2], [1., 2.], ['a', 'b'], [[1, 'a'], [2.]]],
            'c' : [{}, {'a':1, 'b': 2}, {'a':1., 2:3.}],
        },
        invalid = {
            'a' : [],
            'b' : [[1, 'a'], [[], 'a']],
            'c' : [{1.:'a'}, {1:'a', 2:3}],
        },
    )


def test_default():
    """ Test parameter with default value """
    class Test(BaseAbstract):
        def __init__(self, a: int, b: str = 'a'): pass

    with error('a', int, 'a'): Test('a', 1)
    with error('a', int, 'a'): Test('a')
    with error('b', str, 1):   Test(1, 1)
    Test(1, 'a')
    Test(1)


def test_return():
    """ Test return types """
    run(T       = {'return': int}, 
        valid   = {'return': [1]},
        invalid = {'return': ['a']})

    run(T       = {'return': list[int]}, 
        valid   = {'return': [[1], []]},
        invalid = {'return': [['a'], [[1]], None]})

    class ReturnClass: pass
    T = TypeVar('T')
    R = ReturnClass()
    run(T       = {'return': dict[str | int, ('ReturnClass', T)]}, 
        valid   = {'return': [{'a': (R, 1), 2: (R, 3)}, {'b': (R, 1.)}]},
        invalid = {'return': [{'a': (R, 1), 2: (R, 3.)}, {'b': (R,)},
                              { 2.: (R, 1)}, {1: (R, [1]), 'b': (R, 1)}]})


def test_abstractmethod():
    """ Verify abstractmethods are enforced on inheriting classes """
    class Parent(BaseAbstract):
        @abstractmethod
        def test(self): pass

    class Child(Parent): pass
    with pytest.raises(TypeError):
        Child()
