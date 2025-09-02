import pytest
import xarray as xr 
import numpy as np 

from crest.data.transform import Transform


stats = ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75']
example_stats = xr.DataArray([ 
    # mean     std   min   max  median  p25  p75
    [  5.5,      2,   -3,   20,      6,   2,   8],
    [  -30, 10.123,  -42, 1000,     65,   0, 321],
], coords={'features': ['feat_1', 'feat_2'], 'statistics': stats})

example_stats_pos = xr.DataArray([
    # mean     std   min   max  median  p25  p75
    [  5.5,      2,   3,    20,      6,   1,   8],
    [   30, 10.123, 0.1,  1000,     65,   1, 321],
], coords={'features': ['feat_1', 'feat_2'], 'statistics': stats})

example_data = {
    'feat_1': np.atleast_1d([2, 6, -1, 0.5]),
    'feat_2': np.atleast_2d([-20, 3, 999, 0.0123]),
}

example_data_pos = {
    'feat_1': np.atleast_1d([3.001, 4.1, 10, 20]),
    'feat_2': np.atleast_2d([0.1, 0.101, 999, 100.0123]),
}

all_examples = [
    (example_data, example_stats),
    (example_data_pos, example_stats_pos)
]



def check_equality(transform, data, transformed, tol=1e-5):
    """ Verify that `data == inverse(forward(data))` """
    for k in data:

        # Handle negatives in the log transforms
        if np.isnan(transformed[k]).any():
            pos_data = data[k][data[k] > 0]
            pos_tran = transformed[k][data[k] > 0]
            err_vals = [transform, pos_data, pos_tran]
            assert(np.isclose(pos_data, pos_tran, atol=tol).all()), err_vals

            neg_data = data[k][data[k] <= 0]
            neg_tran = transformed[k][data[k] <= 0]
            is_close = np.isclose(neg_data, neg_tran, atol=tol)
            err_vals = [transform, neg_data, neg_tran]
            assert((is_close | np.isnan(neg_tran)).all()), err_vals
            continue

        err_vals = [transform, data[k], transformed[k]]
        assert(np.isclose(data[k], transformed[k], atol=tol).all()), err_vals


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.skip_on_fail
@pytest.mark.parametrize('datastats', all_examples)
@pytest.mark.parametrize('transform', Transform.available_transforms)
def test_symmetry_single(datastats, transform):
    """ Verify forward and inverse are symmetric for all transforms """
    if transform != 'combination':
        data, stats = datastats
        transformer = Transform(stats, by_feature={
            '*'           : 'robust',
            list(data)[0] : 'normalize',
        })

        forward, inverse = getattr(transformer, transform)
        transformed_data = inverse(forward(data))
        check_equality(transform, data, transformed_data)


@pytest.mark.skip_on_fail
@pytest.mark.parametrize('datastats', all_examples)
@pytest.mark.parametrize('transform', Transform.available_transforms)
@pytest.mark.parametrize('transform2', Transform.available_transforms)
def test_symmetry_combination(datastats, transform, transform2):
    """ Verify forward and inverse are symmetric for all combinations """
    if ((transform  not in ['combination', transform2]) and
        (transform2 not in ['combination', transform])):

        # Any log transform with negatives can produce all NaNs in combination
        if any('logexp' in t for t in [transform, transform2]):
            if any((d <= 0).any() for d in datastats[0].values()):
                return 

        c1 = f'robust,{transform}'.replace(',robust','').replace(',by_feature','')
        c2 = f'robust,{transform2}'.replace(',robust','').replace(',by_feature','')

        # Log transforms must be applied first to avoid negatives
        if 'logexp' in transform:  c1 = ','.join(c1.split(',')[::-1])
        if 'logexp' in transform2: c2 = ','.join(c2.split(',')[::-1])

        by_feature = 'by_feature' in [transform, transform2]
        
        # Log transforms cannot be applied second
        if not by_feature and ('logexp' in transform2):
            return 

        data, stats = datastats
        transformer = Transform(stats, by_feature={'*': c1, list(data)[0]: c2})

        if by_feature:
            forward, inverse = transformer.by_feature
        else:
            forward, inverse = transformer.combination([transform, transform2])
        
        transformed_data = inverse(forward(data))
        check_equality([transform, transform2], data, transformed_data)


@pytest.mark.parametrize('datastats', all_examples)
def test_drop(datastats):
    """ Verify that dropped features aren't transformed """
    data, stats = datastats
    dropped_key = list(data)[0]
    transformer = Transform(stats, drop=[dropped_key])

    forward, inverse = transformer.normalize
    transformed_data = forward(data)
    assert((data[dropped_key] == transformed_data[dropped_key]).all())
    transformed_data = inverse(transformed_data)
    assert((data[dropped_key] == transformed_data[dropped_key]).all())


@pytest.mark.parametrize('datastats', all_examples)
def test_by_parameter(datastats):
    """ Test setting transform parameters per feature """
    data, stats = datastats

    # Place min and max at index 0 and 1
    for k in data:
        data[k][..., 0] = stats.sel(features=k, statistics='min').item()
        data[k][..., 1] = stats.sel(features=k, statistics='max').item()

    # The final transform is normalize with set parameters
    key1, *keys = list(data)
    transformer = Transform(stats, by_feature={
        '*'  : 'standardize,robust,normalize|minim=-5|maxim=-2',
        key1 : 'robust,normalize|minim=10|maxim=20'
    })

    forward, inverse = transformer.by_feature
    forward_data = forward(data)
    inverse_data = inverse(forward_data)
    check_equality('by_parameter', data, inverse_data)

    # Verify min/max were set according to their respective parameter
    assert(np.isclose(forward_data[key1][..., 0], 10).all()),forward_data[key1]
    assert(np.isclose(forward_data[key1][..., 1], 20).all()),forward_data[key1]

    for k in keys:
        assert(np.isclose(forward_data[k][..., 0], -5).all()), forward_data[k]
        assert(np.isclose(forward_data[k][..., 1], -2).all()), forward_data[k]
