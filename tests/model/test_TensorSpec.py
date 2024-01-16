import os
from crest.model.TensorSpec import TensorSpec

def test_DictSpec():
    spec = TensorSpec({'shape': (1, 2, 3), 'dtype': 'float32', 'name': 'test'})

    print(spec.shape)

    assert spec.shape == (1, 2, 3)
    assert spec.dtype == 'float32'
    assert spec.name == 'test'

    print(spec.specs)

    assert spec.specs == {'shape': (1, 2, 3), 'dtype': 'float32', 'name': 'test'}

    keras_spec = spec.keras

    assert keras_spec.shape == (1, 2, 3)
    assert keras_spec.dtype == 'float32'
    assert keras_spec.name == 'test'

def test_TupleSpec():
    spec = TensorSpec((1, 2, 3))

    assert spec.shape == (1, 2, 3)
    assert spec.dtype == 'float32'
    assert spec.name == None
    assert spec.specs == {'shape': (1, 2, 3), 'dtype': 'float32', 'name': None}

    keras_spec = spec.keras

    assert keras_spec.shape == (1, 2, 3)
    assert keras_spec.dtype == 'float32'
    assert keras_spec.name == None

def test_ListSpec():
    spec = TensorSpec([1, 2, 3])

    assert spec.shape == (1, 2, 3)
    assert spec.dtype == 'float32'
    assert spec.name == None
    assert spec.specs == {'shape': [1, 2, 3], 'dtype': 'float32', 'name': None}

    keras_spec = spec.keras

    assert keras_spec.shape == (1, 2, 3)
    assert keras_spec.dtype == 'float32'
    assert keras_spec.name == None

def test_TupleSpecWithDtype():
    spec = TensorSpec((1, 2, 3), 'int32')

    assert spec.shape == (1, 2, 3)
    assert spec.dtype == 'int32'
    assert spec.name == None
    assert spec.specs == {'shape': (1, 2, 3), 'dtype': 'int32', 'name': None}

    keras_spec = spec.keras

    assert keras_spec.shape == (1, 2, 3)
    assert keras_spec.dtype == 'int32'
    assert keras_spec.name == None

def test_TupleSpecWithName():
    spec = TensorSpec((1, 2, 3), 'int32', 'test')

    assert spec.shape == (1, 2, 3)
    assert spec.dtype == 'int32'
    assert spec.name == 'test'
    assert spec.specs == {'shape': (1, 2, 3), 'dtype': 'int32', 'name': 'test'}

    keras_spec = spec.keras

    assert keras_spec.shape == (1, 2, 3)
    assert keras_spec.dtype == 'int32'
    assert keras_spec.name == 'test'

def test_save_load():
    spec = TensorSpec({'shape': (1, 2, 3), 'dtype': 'float32', 'name': 'test'})
    spec.save('test-tensorspec.json')

    assert os.path.exists('test-tensorspec.json')

    loaded_spec = TensorSpec.load('test-tensorspec.json')

    assert spec == loaded_spec

    os.remove('test-tensorspec.json')