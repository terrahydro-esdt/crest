from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import warnings
import pytest
import json

from crest.utils import json_safe


@pytest.mark.parametrize(
    'obj, expected',
    [
        (None, None),
        (True, True),
        (False, False),
        (0, 0),
        (1, 1),
        (-1, -1),
        (1.5, 1.5),
        ('abc', 'abc'),
        ('', ''),
        ([1, 'two', None, True], [1, 'two', None, True]),
        (('a', 1, None), ['a', 1, None]),
        (
            {'a': 1, 'b': [2, 3, {'c': ('x', 'y')}]},
            {'a': 1, 'b': [2, 3, {'c': ['x', 'y']}]},
        ),
        (
            defaultdict(int, {'a': 1, 'b': {'c': 2}}),
            {'a': 1, 'b': {'c': 2}},
        ),
    ],
)
def test_no_warnings(obj, expected):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = json_safe(obj)

    assert result == expected
    assert(len(record) == 0), record
    json.dumps(result)


@pytest.mark.parametrize(
    'obj, expected',
    [
        ({'path': Path('/tmp/a.nc')}, {'path': '/tmp/a.nc'}),
        (
            {'created': datetime(2026, 5, 14, 12, 30)},
            {'created': '2026-05-14 12:30:00'},
        ),
        (
            {'array': np.array([[1, 2], [3, 4]])},
            {'array': str(np.array([[1, 2], [3, 4]]))},
        ),
        (
            {'value': np.float32(1.5)},
            {'value': str(np.float32(1.5))},
        ),
    ],
)
def test_warnings(obj, expected):
    with pytest.warns(UserWarning):
        result = json_safe(obj)

    assert result == expected
    json.dumps(result)


def test_mixed_warnings():
    arr = np.ones((1, 2))
    obj = {
        'mixed_types': [
            0,
            datetime(2026, 5, 14, 12, 30),
            {
                'key': 1,
                'numpy': arr,
            },
        ]
    }

    with pytest.warns(UserWarning) as record:
        result = json_safe(obj)
    for i,r in enumerate(record):
        print(i, r.message)
    assert(len(record) == 2), [r.message for r in record]
    assert result == {
        'mixed_types': [
            0,
            '2026-05-14 12:30:00',
            {
                'key': 1,
                'numpy': str(arr),
            },
        ]
    }
    json.dumps(result)


@pytest.mark.parametrize(
    'obj, expected',
    [
        ({Path('/tmp/key'): 'value'}, {'/tmp/key': 'value'}),
        ({('tuple', 'key'): 'value'}, {"('tuple', 'key')": 'value'}),
    ],
)
def test_non_string_keys(obj, expected):
    with pytest.warns(UserWarning):
        result = json_safe(obj)

    assert result == expected
    json.dumps(result)


def test_original_not_mutated():
    obj = {
        'a': [1, 2, Path('/tmp/file')],
        'b': {'c': datetime(2026, 5, 14)},
    }

    original_a = list(obj['a'])
    original_b = dict(obj['b'])

    with pytest.warns(UserWarning):
        result = json_safe(obj)

    assert obj['a'] == original_a
    assert obj['b'] == original_b

    assert result is not obj
    assert result['a'] is not obj['a']
    assert result['b'] is not obj['b']


def test_json_loadable():
    obj = {
        'title': 'Example',
        'valid_range': [0.0, 1.0],
        'created': datetime(2026, 5, 14),
        'nested': {
            'path': Path('/tmp/example.nc'),
            'shape': (1, 2, 3),
        },
    }

    with pytest.warns(UserWarning):
        safe = json_safe(obj)

    encoded = json.dumps(safe)
    decoded = json.loads(encoded)

    assert decoded == {
        'title': 'Example',
        'valid_range': [0.0, 1.0],
        'created': '2026-05-14 00:00:00',
        'nested': {
            'path': '/tmp/example.nc',
            'shape': [1, 2, 3],
        },
    }