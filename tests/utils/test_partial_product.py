from functools import update_wrapper 
from itertools import product

import numpy as np
import pytest

from crest.src.utils import partial_product 


# Number of times to execute tests on random data 
REPEATS = 5

# Global random seed
SEED = None


def generate_data() -> list[list]:
    """ Generate random number of lists with random length """
    number = 2, 7 # Range for number of lists
    length = 1, 7 # Range for size of each list 
    random = np.random.default_rng(SEED).integers
    create = lambda _: list(range(random(*length)))
    return list(map(create, range(random(*number))))

def check_equals(data: list[list], prod: list[tuple], i: slice = slice(None)):
    """ Check partial==itertools.product, and that slice/arg formats equal """
    partial = list(partial_product(data, i))
    arg_out = list(partial_product(data, i.start, i.stop, i.step))
    assert(partial == arg_out), [partial, arg_out]
    assert(partial == prod[i]), [partial, prod[i], i]

def repeat(function: callable) -> callable: 
    """ Decorator which repeats a function `count` times """
    wrapper = lambda: [function() for _ in range(REPEATS)][-1]
    return update_wrapper(wrapper, function)


@repeat
def test_equals_product():
    """ Simply test equality with itertools.product """
    data = generate_data()
    prod = list(product(*data))
    check_equals(data, prod)


@repeat
def test_start():
    """ Test starting index """
    data = generate_data()
    prod = list(product(*data))

    i = np.random.default_rng(SEED).integers(len(prod))
    check_equals(data, prod, slice(i))


@repeat
def test_stop():
    """ Test stopping index """
    data = generate_data()
    prod = list(product(*data))

    i = np.random.default_rng(SEED).integers(len(prod))
    check_equals(data, prod, slice(None, i))


@repeat
def test_step():
    """ Test stepping index """
    data = generate_data()
    prod = list(product(*data))

    i = np.random.default_rng(SEED).integers(1, max(len(prod), 2))
    check_equals(data, prod, slice(None, None, i))


@repeat
def test_range():
    """ Test starting and stopping index """
    data = generate_data()
    prod = list(product(*data))

    i = np.random.default_rng(SEED).integers(len(prod))
    j = np.random.default_rng(SEED).integers(i, len(prod))
    check_equals(data, prod, slice(i, j))


@repeat
def test_range_and_step():
    """ Test starting, stopping, and stepping index """
    data = generate_data()
    prod = list(product(*data))

    i = np.random.default_rng(SEED).integers(len(prod))
    j = np.random.default_rng(SEED).integers(i, len(prod))
    k = np.random.default_rng(SEED).integers(1, max(j-i, 2))
    check_equals(data, prod, slice(i, j, k))


def test_one_empty():
    """ Test an empty list among the data """
    data = generate_data() + []
    prod = list(product(*data))
    check_equals(data, prod)


def test_only_empty():
    """ Test all empty lists """
    data = [[]] * 5
    prod = list(product(*data))
    check_equals(data, prod)


def test_missing():
    """ Test no data """
    data = []
    prod = list(product(*data))
    check_equals(data, prod)