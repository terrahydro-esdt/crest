import pytest
from crest import DataServer


def test_load_error():
    with pytest.raises(Exception):
        DataServer('foo')
