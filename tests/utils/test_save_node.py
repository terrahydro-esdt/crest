import pytest
import tempfile
import json
import os
from pathlib import Path
from shapely.geometry import Point
from crest.utils.save_node_class import *


class DummyModel:
    def __init__(self):
        self.value = 42
        self.log = "This should be stripped"
        self.temp_cache = {"a": 1}

    def compute(self):
        return self.value + 10


def test_write_and_read_pickle():
    model = DummyModel()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name

    try:
        write_pkl(model, path)
        restored = read_pkl(path, metadata_path=Path(path).parent)

        assert restored.__class__.__name__ == DummyModel.__name__
        assert restored.value == 42
        assert not getattr(restored, "log", None)
        assert not getattr(restored, "temp_cache", None)
    finally:
        os.remove(path)


def test_gen_filename_uniqueness():
    name1 = gen_filename(prefix="test")
    name2 = gen_filename(prefix="test")
    assert name1 != name2
    assert name1.endswith(".pkl")
    assert name2.endswith(".pkl")


def test_get_class_module_path():
    path = get_class_module_path(DummyModel)
    assert path.endswith(".py")
    assert Path(path).exists()
