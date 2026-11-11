import numpy as np
import pytest
import xarray as xr

from crest.data.loading import Sample


# =============================================================================
# Fixtures
# =============================================================================


def _window(
    features           : list[str],
    values             : list[list[float]],
    coords             : dict[str, np.ndarray],
    key_label          : str = '',
) -> dict:
    """Build one raw Datafile window using Block.extract's Sample format."""
    dims = ['features'] + list(coords)
    return {
        'data'               : np.asarray(values),
        'coords'             : {'features': features} | coords,
        'dims'               : dims,
        'key_label'          : key_label,
    }


@pytest.fixture
def sample() -> Sample:
    """Sample with duplicated features and distinct source coordinates."""
    return Sample([
        _window(
            ['a', 'shared'],
            [[1., 2.], [10., 20.]],
            {'x': np.array([0., 1.])},
            key_label='grid/zero',
        ),
        _window(
            ['b', 'shared'],
            [[3., 4.], [30., 40.]],
            {'x': np.array([5., 6.])},
            key_label='grid/one',
        ),
    ])


# =============================================================================
# Collection interface
# =============================================================================


def test_collection_interface(sample: Sample) -> None:
    """Sample behaves as a lazy collection of Datafile xarray windows."""
    assert len(sample) == 2
    assert sample.dtype is object
    assert sample.astype('float32') is sample

    array = np.asarray(sample)
    assert array.shape == (1,)
    assert array.dtype == object
    assert array[0] is sample

    expected = xr.Dataset(
        data_vars={
            'a'      : ('x', [1., 2.]),
            'shared' : ('x', [10., 20.]),
        },
        coords={'x': [0., 1.]},
    )
    xr.testing.assert_equal(sample[0], expected)
    xr.testing.assert_equal(sample['a'], expected)
    xr.testing.assert_equal(list(sample)[0], expected)



def test_getitem_caches_loaded_dataset(
    sample      : Sample,
    monkeypatch : pytest.MonkeyPatch,
) -> None:
    """Repeated indexing must not reconstruct an already loaded Dataset."""
    original = sample._load
    calls    = []

    def tracked(idx: int) -> xr.Dataset:
        calls.append(idx)
        return original(idx)

    monkeypatch.setattr(sample, '_load', tracked)
    first  = sample[0]
    second = sample[0]

    assert first is second
    assert calls == [0]



def test_index_uses_first_physical_source(sample: Sample) -> None:
    """Feature lookup identifies the first Datafile containing the feature."""
    assert sample.index('a') == 0
    assert sample.index('b') == 1
    assert sample.index('shared') == 0

    with pytest.raises(ValueError, match='missing'):
        sample.index('missing')



def test_nbytes_counts_all_loaded_windows(sample: Sample) -> None:
    """Reported size is the total size of all Datafile Dataset windows."""
    expected = sample[0].nbytes + sample[1].nbytes
    assert sample.nbytes == expected


# =============================================================================
# Raw views
# =============================================================================


def test_dimensions_and_features(sample: Sample) -> None:
    """Dimension and feature views preserve source ordering."""
    assert sample.dims == ['features', 'x']
    assert sample.features == ['a', 'shared', 'b', 'shared']



def test_coordinate_views(sample: Sample) -> None:
    """Coordinates remain grouped by coordinate name and source order."""
    coords = sample.coords
    np.testing.assert_array_equal(coords['features'][0], ['a', 'shared'])
    np.testing.assert_array_equal(coords['features'][1], ['b', 'shared'])
    np.testing.assert_array_equal(coords['x'][0], [0., 1.])
    np.testing.assert_array_equal(coords['x'][1], [5., 6.])

    assert sample.coords_avg['x'] == pytest.approx(3.)



def test_data_view_preserves_duplicate_sources(sample: Sample) -> None:
    """Data view maps duplicate feature names back to their source indices."""
    data = sample.data
    np.testing.assert_array_equal(data['a'][0], [1., 2.])
    np.testing.assert_array_equal(data['b'][1], [3., 4.])
    np.testing.assert_array_equal(data['shared'][0], [10., 20.])
    np.testing.assert_array_equal(data['shared'][1], [30., 40.])
    np.testing.assert_array_equal(data['x'][0], [0., 1.])
    np.testing.assert_array_equal(data['x'][1], [5., 6.])



def test_data_groups_preserve_datafile_boundaries(sample: Sample) -> None:
    """Grouped data combines each source's features and coordinates."""
    groups = sample.data_groups
    assert len(groups) == 2

    np.testing.assert_array_equal(groups[0]['a'], [1., 2.])
    np.testing.assert_array_equal(groups[0]['shared'], [10., 20.])
    np.testing.assert_array_equal(groups[0]['x'], [0., 1.])
    np.testing.assert_array_equal(groups[1]['b'], [3., 4.])
    np.testing.assert_array_equal(groups[1]['shared'], [30., 40.])
    np.testing.assert_array_equal(groups[1]['x'], [5., 6.])



def test_get_feature_coords(sample: Sample) -> None:
    """Feature coordinates come from the first source containing the feature."""
    coords = sample.get_feature_coords('shared')
    assert set(coords) == {'x'}
    np.testing.assert_array_equal(coords['x'], [0., 1.])


# =============================================================================
# Feature selection
# =============================================================================


def test_plain_features_use_first_physical_source(sample: Sample) -> None:
    """Unqualified features use the first physical source containing them."""
    a, b, shared = sample.to_list(['a', 'b', 'shared'])
    np.testing.assert_array_equal(a, [1., 2.])
    np.testing.assert_array_equal(b, [3., 4.])
    np.testing.assert_array_equal(shared, [10., 20.])



def test_default_feature_selection_preserves_occurrences(sample: Sample) -> None:
    """Default list output retains every feature occurrence in source order."""
    values = sample.to_list()
    expected = [
        [1., 2.],
        [10., 20.],
        [3., 4.],
        [30., 40.],
    ]

    assert len(values) == len(sample.features)
    for output, target in zip(values, expected):
        np.testing.assert_array_equal(output, target)



def test_explicit_source_index(sample: Sample) -> None:
    """The @ syntax selects a specific source by index or key label."""
    values = sample.to_list([
        'shared@0',
        'shared@1',
        'shared@grid$zero',
        'shared@grid$one',
    ])
    expected = [
        [10., 20.],
        [30., 40.],
        [10., 20.],
        [30., 40.],
    ]

    for output, target in zip(values, expected):
        np.testing.assert_array_equal(output, target)



def test_explicit_coordinate_source(sample: Sample) -> None:
    """Coordinate names can use the same @ source-selection syntax."""
    x0, x1 = sample.to_list(['x@0', 'x@grid$one'])
    np.testing.assert_array_equal(x0, [0., 1.])
    np.testing.assert_array_equal(x1, [5., 6.])



def test_double_underscore_source_selection(sample: Sample) -> None:
    """The label__feature syntax selects a feature from a named source."""
    zero, one = sample.to_list([
        'grid/zero__shared',
        'grid/one__shared',
    ])
    np.testing.assert_array_equal(zero, [10., 20.])
    np.testing.assert_array_equal(one, [30., 40.])



def test_unknown_double_underscore_label_raises(sample: Sample) -> None:
    """An explicit source selector must refer to a known Datafile."""
    with pytest.raises(ValueError, match='Unknown key/format'):
        sample.to_list(['unknown__shared'])



def test_missing_coordinate_returns_typed_nan() -> None:
    """Selecting a missing source coordinate returns a typed placeholder."""
    sample = Sample([
        _window(
            ['a'], [[1., 2.]],
            {'datetime': np.array(['2020-01-01', '2020-01-02'],
                                  dtype='datetime64[D]')},
            key_label='time',
        ),
        _window(
            ['b'], [[3., 4.]],
            {'x': np.array([5., 6.])},
            key_label='space',
        ),
    ])

    datetime, x = sample.to_list(['datetime@space', 'x@time'])
    assert datetime.shape == (1,)
    assert np.issubdtype(datetime.dtype, np.datetime64)
    assert np.isnat(datetime[0])
    np.testing.assert_array_equal(x, [np.nan])



def test_distance_selector_by_label_and_index(sample: Sample) -> None:
    """Distance selectors return pairwise coordinate differences."""
    by_label, by_index = sample.to_list([
        'distance:grid$zero:grid$one',
        'distance:0:1',
    ])
    expected = np.array([[-5., -6.], [-4., -5.]])

    assert set(by_label) == {'x'}
    np.testing.assert_array_equal(by_label['x'], expected)
    np.testing.assert_array_equal(by_index['x'], expected)



def test_invalid_distance_selector(sample: Sample) -> None:
    """Malformed and same-source distance requests are rejected."""
    with pytest.raises(ValueError, match='expected format'):
        sample.to_list(['distance:0'])

    with pytest.raises((AssertionError, ValueError)):
        sample.to_list(['distance:0:0'])



def test_unknown_source_raises_value_error(sample: Sample) -> None:
    """Explicit source selection rejects unknown source labels."""
    with pytest.raises(ValueError, match='Unknown key/format'):
        sample.to_list(['shared@unknown'])


def test_unavailable_feature_from_known_source_raises(sample: Sample) -> None:
    """Missing data features do not acquire coordinate placeholders."""
    with pytest.raises(ValueError, match='Value "b" not found'):
        sample.to_list(['b@0'])


@pytest.mark.parametrize('selector', [
    'grid/source__b', 'b@grid/source',
    'grid$source__b', 'b@grid$source',
])
def test_labeled_feature_uses_matching_window(selector: str) -> None:
    """A shared source label resolves to the window containing the feature."""
    sample = Sample([
        _window([feature], [values], {'time': np.arange(len(values))}, label)
        for label, feature, values in [
            ('other',       'b', [99.]),
            ('grid/source', 'a', [1., 2.]),
            ('grid/source', 'b', [3., 4., 5.]),
        ]
    ])

    value, = sample.to_list([selector])
    np.testing.assert_array_equal(value, [3., 4., 5.])

    # Numeric selectors must still select that exact window
    with pytest.raises(ValueError, match='Value "b" not found'):
        sample.to_list(['b@1'])

    # Feature selection must not construct xarray windows
    assert sample.cache == {}


# =============================================================================
# Output formats
# =============================================================================


def test_empty_feature_selection_is_empty(sample: Sample) -> None:
    """An explicit empty feature list requests no output features."""
    assert sample.to_list([]) == []
    assert sample.to_array([]).shape == (0,)
    assert sample.to_dict([]) == {}



def test_to_array(sample: Sample) -> None:
    """Selected equally shaped features stack into a numpy array."""
    output = sample.to_array(['a', 'b', 'shared@1'])
    expected = np.array([
        [1., 2.],
        [3., 4.],
        [30., 40.],
    ])
    np.testing.assert_array_equal(output, expected)



def test_to_dict(sample: Sample) -> None:
    """Dictionary output retains selector ordering and resolved values."""
    output = sample.to_dict(['b', 'shared', 'shared@1', 'a'])
    assert list(output) == ['b', 'shared', 'shared@1', 'a']
    np.testing.assert_array_equal(output['b'], [3., 4.])
    np.testing.assert_array_equal(output['shared'], [10., 20.])
    np.testing.assert_array_equal(output['shared@1'], [30., 40.])
    np.testing.assert_array_equal(output['a'], [1., 2.])
