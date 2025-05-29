import pytest
import xarray as xr
import tensorflow as tf
import os, json

from crest.utils import Metrics


def sample_simple_data():
    observed_data = [float(x) for x in range(365)]
    simulated_data = [float(x) for x in range(365)]

    # create a DataArray with the observed and simulated data
    observed_array = xr.DataArray(
        observed_data,
        coords={"time": range(365)},
        dims=["time"]
    )

    simulated_array = xr.DataArray(
        simulated_data,
        coords={"time": range(365)},
        dims=["time"]
    )

    obs = tf.convert_to_tensor(observed_data)
    pred = tf.convert_to_tensor(simulated_data)

    return observed_array, simulated_array, obs, pred


def sample_complex_data():
    observed_data = [[float(x) + 1, float(x)] for x in range(10)]
    simulated_data = [[float(x)*0.05, float(x)*0.1] for x in range(10)]

    # create a DataArray with the observed and simulated data
    observed_array = xr.DataArray(
        observed_data,
        dims=['x', 'y']
    )

    simulated_array = xr.DataArray(
        simulated_data,
        dims=['x', 'y']
    )

    obs = tf.convert_to_tensor(observed_data)
    pred = tf.convert_to_tensor(simulated_data)

    return observed_array, simulated_array, obs, pred


def test_relative_error():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.relative_error(observed, predicted)

    assert cmetrics.numpy() == 0.0


def test_relative_error_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.relative_error(observed, predicted)

    assert cmetrics.numpy() == pytest.approx(-0.8873224, 1e-5)


def test_metric_entropy():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    results = mcrest.metric_entropy(observed, predicted)

    assert (results[0].numpy() == pytest.approx(0.0136986, 1e-5)
            ) and (results[1].numpy() == pytest.approx(0.0136986, 1e-5))


def test_metric_entropy_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    results = mcrest.metric_entropy(observed, predicted)

    result_0 = (tf.reduce_mean(results[0].numpy())).numpy()
    result_1 = (tf.reduce_mean(results[1].numpy())).numpy()

    assert (result_0 == 0.1 and result_1 == 0.1)


def test_nse():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.nse(observed, predicted)

    assert cmetrics == 1.0


def test_nse_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.nse(observed, predicted)

    assert cmetrics == pytest.approx(-2.432720, 1e-5)


def test_mse():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.mse(observed, predicted)

    assert cmetrics.numpy() == 0.0


def test_mse_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.mse(observed, predicted)

    assert cmetrics.numpy() == pytest.approx(29.178125, 1e-5)


def test_rmse():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.unbiased_rmse(observed, predicted)

    assert cmetrics == 0.0


def test_rmse_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.unbiased_rmse(observed, predicted)

    assert cmetrics == pytest.approx(5.401677, 1e-5)


def test_alpha_nse():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.alpha_nse(observed, predicted)

    assert cmetrics == 1.0


def test_alpha_nse_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.alpha_nse(observed, predicted)

    assert cmetrics == pytest.approx(0.0869203, 1e-5)


def test_beta_nse():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.beta_nse(observed, predicted)

    assert cmetrics == 0.0


def test_beta_nse_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.beta_nse(observed, predicted)

    assert cmetrics == pytest.approx(-1.599224, 1e-5)


def test_kge():
    observed, predicted, obs, pred = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.kge(observed, predicted)

    assert cmetrics == 1.0


def test_pearsonr():
    observed, predicted, obs, pred = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.pearsonr(obs, pred)

    assert cmetrics == pytest.approx(1.0)


def test_call_keras_metrics():
    _, _, obs, pred = sample_simple_data()

    mcrest = Metrics()
    accuracy = mcrest.get_callbacks('accuracy')
    func = accuracy[0]
    cmetrics = func(obs, pred)

    assert len(cmetrics.numpy()) > 1


def test_call_keras_metrics_complex():
    _, _, obs, pred = sample_complex_data()

    mcrest = Metrics()
    accuracy = mcrest.get_callbacks('accuracy')
    func = accuracy[0]
    cmetrics = func(obs, pred)

    assert len(cmetrics.numpy()) > 1

def test_save_load():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.nse(observed, predicted)
    saved_dict = mcrest.to_json()
    mcrest2 = Metrics.from_json(saved_dict)
    cmetrics2 = mcrest2.nse(observed, predicted)

    assert cmetrics == cmetrics2

def test_save_load_file():
    observed, predicted, _, _ = sample_simple_data()

    mcrest = Metrics()
    cmetrics = mcrest.nse(observed, predicted)
    saved_dict = mcrest.to_json()

    with open('test.json', 'w') as f:
        f.write(saved_dict)

    with open('test.json', 'r') as f:
        saved_dict = f.read()
        mcrest2 = Metrics.from_json(saved_dict)
        cmetrics2 = mcrest2.nse(observed, predicted)

        assert cmetrics == cmetrics2

    os.remove('test.json')

def test_save_load_complex():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.nse(observed, predicted)
    saved_dict = mcrest.to_json()
    mcrest2 = Metrics.from_json(saved_dict)
    cmetrics2 = mcrest2.nse(observed, predicted)

    assert cmetrics == cmetrics2

def test_save_load_complex_file():
    observed, predicted, _, _ = sample_complex_data()

    mcrest = Metrics()
    cmetrics = mcrest.nse(observed, predicted)
    saved_dict = mcrest.to_json()

    with open('test.json', 'w') as f:
        f.write(saved_dict)

    with open('test.json', 'r') as f:
        saved_dict = f.read()
        mcrest2 = Metrics.from_json(saved_dict)
        cmetrics2 = mcrest2.nse(observed, predicted)

        assert cmetrics == cmetrics2

    os.remove('test.json')

# test subclasses of Metrics
def test_relative_error_subclass():
    observed, predicted, _, _ = sample_simple_data()

    cmetrics = Metrics.RelativeError().update_state(observed, predicted)

    assert cmetrics.numpy() == 0.0

def test_relative_error_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    cmetrics = Metrics.RelativeError().update_state(observed, predicted)

    assert cmetrics.numpy() == pytest.approx(-0.8873224, 1e-5)

def test_metric_entropy_subclass():
    observed, predicted, _, _ = sample_simple_data()

    results = Metrics.MetricEntropy().update_state(observed, predicted)

    assert (results[0].numpy() == pytest.approx(0.0136986, 1e-5)
            ) and (results[1].numpy() == pytest.approx(0.0136986, 1e-5))
    
def test_metric_entropy_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    results = Metrics.MetricEntropy().update_state(observed, predicted)

    result_0 = (tf.reduce_mean(results[0].numpy())).numpy()
    result_1 = (tf.reduce_mean(results[1].numpy())).numpy()

    assert (result_0 == 0.1 and result_1 == 0.1)

def test_nse_subclass():
    observed, predicted, _, _ = sample_simple_data()

    cmetrics = Metrics.NSE().update_state(observed, predicted)

    assert cmetrics == 1.0

def test_nse_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    cmetrics = Metrics.NSE().update_state(observed, predicted)

    print(cmetrics)

    assert cmetrics.numpy() == pytest.approx(-2.432720, 1e-5)

def test_mse_subclass():
    observed, predicted, _, _ = sample_simple_data()

    cmetrics = Metrics.MSE().update_state(observed, predicted)

    assert cmetrics.numpy() == 0.0

def test_mse_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    cmetrics = Metrics.MSE().update_state(observed, predicted)

    assert cmetrics.numpy() == pytest.approx(29.178125, 1e-5)

def test_rmse_subclass():
    observed, predicted, _, _ = sample_simple_data()

    cmetrics = Metrics.UnbiasedRMSE().update_state(observed, predicted)

    assert cmetrics == 0.0

def test_rmse_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    cmetrics = Metrics.UnbiasedRMSE().update_state(observed, predicted)

    assert cmetrics == pytest.approx(5.401677, 1e-5)

def test_alpha_nse_subclass():
    observed, predicted, _, _ = sample_simple_data()

    cmetrics = Metrics.AlphaNSE().update_state(observed, predicted)

    assert cmetrics == 1.0

def test_alpha_nse_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    alphanse = Metrics.AlphaNSE()
    cmetrics = alphanse.update_state(observed, predicted)

    assert cmetrics.numpy() == pytest.approx(0.0869203, 1e-5)

def test_beta_nse_subclass():
    observed, predicted, _, _ = sample_simple_data()

    betanse = Metrics.BetaNSE()
    cmetrics = betanse.update_state(observed, predicted)

    assert cmetrics == 0.0

def test_beta_nse_subclass_complex():
    observed, predicted, _, _ = sample_complex_data()

    betanse = Metrics.BetaNSE()
    cmetrics = betanse.update_state(observed, predicted)

    assert cmetrics.numpy() == pytest.approx(-1.599224, 1e-5)

def test_kge_subclass():
    observed, predicted, _, _ = sample_simple_data()

    kge = Metrics.KGE()
    cmetrics = kge.update_state(observed, predicted)

    assert cmetrics == 1.0

def test_pearsonr_subclass():
    _, _, obs, pred = sample_simple_data()

    pearsonr = Metrics.Pearson()
    cmetrics = pearsonr.update_state(obs, pred)

    assert cmetrics == pytest.approx(1.0)
