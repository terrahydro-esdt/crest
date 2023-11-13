import pytest
import numpy as np 

from crest.crest.base.BaseSet import BaseSet


intset  = BaseSet([1,2,3,6,5,4])
strset  = BaseSet(['a b c', 'd.e.f'])


def test_len():
    assert(len(intset) == 6)
    assert(len(strset) == 2)


def test_iter():
    sets = [
        (intset, [1, 2, 3, 6, 5, 4]),
        (strset, ['a b c', 'd.e.f']),
    ]
    for baseset, expected in sets:
        for b, e in zip(baseset, expected):
            assert(b == e)


def test_getattr():
    # Test mapping a function with no arguments
    output = intset.bit_length()
    expect = [1, 2, 2, 3, 3, 3]
    assert(output == expect)

    output = strset.upper()
    expect = ['A B C', 'D.E.F']
    assert(output == expect)

    # Test mapping a function with an argument
    output = intset.__add__(1)
    expect = [2, 3, 4, 7, 6, 5]
    assert(output == expect)

    output = strset.join(['1 ', ' 2'])
    expect = ['1 a b c 2', '1 d.e.f 2']
    assert(output == expect)

    # Test mapping a function with one argument per element
    output = intset + [0, 1, 2, 0, 0, 0]
    expect = [1, 3, 5, 6, 5, 4]
    assert(output == expect)

    output = strset.split(_map=[[' ', '.']])
    expect = [['a', 'b', 'c'], ['d', 'e', 'f']]
    assert(output == expect)

    # Test mapping a function with multiple arguments per element
    output = strset.split(_map=[[' ', '.'], [1, 2]])
    expect = [['a', 'b c'], ['d', 'e', 'f']]
    assert(output == expect)

    # Test mapping with arguments and keywords
    output = strset.split(_map=[[' ', '.']], _kwmap={'maxsplit': [1, 2]})
    expect = [['a', 'b c'], ['d', 'e', 'f']]
    assert(output == expect)


def test_chain():
    # Test chaining functions together
    output = (intset + 1) * [2,1,0,0,1,2]
    expect = [4, 3, 0, 0, 6, 10]
    assert(output == expect)

    output = strset.upper().split(maxsplit=1, _map=[[' ', '.']])
    expect = [['A', 'B C'], ['D', 'E.F']]
    assert(output == expect)


def test_numpy():
    # Ensure numpy compatibility
    output = np.max(intset)
    expect = 6
    assert(output == expect)


def test_sort():
    ordered = lambda i: i 
    reverse = lambda i: -i 

    # Test sorting itself
    container = intset.sort(ordered)
    assert(container == [1,2,3,4,5,6])
    assert(intset.container == container)
    assert(intset.order == [0,1,2,5,4,3])

    # Test sorting another container (and numpy array)
    another   = [10,9,8,5,6,7]
    container = intset.sort(ordered, np.array(another))
    assert(container == [5,6,7,8,9,10])
    assert(intset.container == [1,2,3,4,5,6])
    assert(intset.order == [0,1,2,5,4,3])

    # Test sorting another container using itself as the order
    container = intset.sort(container=another)
    assert(container == [10,9,8,7,6,5])
    assert(intset.container == [1,2,3,4,5,6])

    # Reverse the order - order should NOT be equal to [5,4,3,2,1,0],
    # which is the case if we didn't first restore the original ordering
    container = intset.sort(reverse)
    assert(container == [6,5,4,3,2,1])
    assert(intset.container == container)
    assert(intset.order == [3,4,5,2,1,0])

    # Double check we get the original data back if we 
    # sort using the new order
    container = intset.sort(container=[6,5,4,3,2,1])
    assert(container == [1,2,3,6,5,4])

    # Return to the original order
    container = intset.sort()
    assert(container == [1,2,3,6,5,4])
    assert(intset.container == container)
    assert(intset.order == [0,1,2,3,4,5])

    # No change, since it uses the original ordering
    container = intset.sort(container=another)
    assert(container == another)


def test_homogeneity():
    with pytest.raises(TypeError): container = BaseSet([1,2,'a'])
    with pytest.raises(TypeError): container = BaseSet([[], 'a'])
    
    # BaseSet only requires homogeneity on the first container level
    container = BaseSet([[1], ['a']])
