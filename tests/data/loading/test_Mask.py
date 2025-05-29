import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from unittest.mock import patch, MagicMock
from pathlib import Path
from types import SimpleNamespace
from crest.data.loading.RegionalMaskGenerator import RegionalMaskGenerator


# Sample GeoDataFrame to simulate geocoded regions
def sample_gdf():
    return gpd.GeoDataFrame({
        'name': ['Region1'],
        'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    }, geometry='geometry')


# Sample data object that mimics xarray.DataArray with latitude and longitude
@pytest.fixture
def sample_xarray():
    lat = np.linspace(0, 1, 100)
    lon = np.linspace(0, 1, 100)
    data = xr.DataArray(np.random.rand(100, 100), coords=[
                        ('latitude', lat), ('longitude', lon)])
    return data


@patch('osmnx.geocode_to_gdf')
def test_load_by_name(mock_geocode):
    mock_geocode.return_value = sample_gdf()
    gen = RegionalMaskGenerator(['Region1'])
    assert isinstance(gen.gdf, gpd.GeoDataFrame)
    assert not gen.gdf.empty


@patch('osmnx.geocode_to_gdf')
def test_gen_mask(mock_geocode, sample_xarray):
    mock_geocode.return_value = sample_gdf()
    gen = RegionalMaskGenerator(['Region1'])
    success = gen.gen_mask(sample_xarray)
    assert not success is None
    assert isinstance(gen.mask, np.ndarray)
    assert gen.mask.shape == (100, 100)


@patch('osmnx.geocode_to_gdf')
def test_gen_mask_boundary(mock_geocode):
    mock_geocode.return_value = sample_gdf()
    gen = RegionalMaskGenerator(['Region1'])
    success = gen.gen_mask_boundary(100, 100)
    assert not success is None
    assert isinstance(gen.mask, np.ndarray)
    assert gen.mask.shape == (100, 100)


@patch('osmnx.geocode_to_gdf')
def test_visualize_mask_raises_error_without_mask(mock_geocode):
    mock_geocode.return_value = sample_gdf()
    gen = RegionalMaskGenerator(['Region1'])
    with pytest.raises(ValueError, match="Mask needs to be initialized"):
        gen.visualize_mask()


@patch('osmnx.geocode_to_gdf')
def test_save_mask_raises_error_without_mask(mock_geocode, tmp_path):
    mock_geocode.return_value = sample_gdf()
    gen = RegionalMaskGenerator(['Region1'])
    with pytest.raises(ValueError, match="Mask needs to be initialized"):
        gen.save_mask(tmp_path / "test.tif")
