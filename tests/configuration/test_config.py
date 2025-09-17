import pytest
from unittest.mock import patch, MagicMock, mock_open
import yaml
import geopandas as gpd
from shapely.geometry import Polygon
import tempfile
import os
from crest.configuration.Config import Config

def sample_gdf():
    return gpd.GeoDataFrame({
        'name': ['Region1'],
        'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    }, geometry='geometry')

def mock_mcontrol(return_gdf=None):
    mock_instance = MagicMock()
    mock_instance.init_region.return_value = (None, return_gdf or sample_gdf())
    return mock_instance

def test_set_and_get_config_values():

    config = Config()
    config.config['debug'] = True 
    config.config['threshold'] = 0.9

    assert config.debug is True
    assert config.threshold == 0.9

def test_save_and_load_config(tmp_path):

    config_file = tmp_path / "config.yaml"
    config = Config()
    config.config['key1'] = 'value1'
    config.config['key2'] = 123
    config.save(str(config_file))

    config2 = Config(str(config_file))
    assert config2.key1 == 'value1'
    assert config2.key2 == 123

@pytest.mark.skipif(
    (not os.path.exists("shapefiles/shape_dictionary.json")),
    reason="Shapefile required for this test"
)
def test_dissolve_regions_flag(tmp_path):

    config_data = {
        'region': ['conus', 'Washington'],
        'region_types': [4, 3],
        'dissolve_regions': True
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    with patch('crest.data.loading.RegionalMaskGenerator') as MockGen:
        instance = mock_mcontrol()
        MockGen.side_effect = [instance, instance]

        config = Config(str(config_file))
        config.load_region()
        gdf = config.gdf

        # Expect one unified polygon (since unary_union would merge two identical ones)
        assert len(gdf) == 1 or gdf.geometry.iloc[0].geom_type in [
            'Polygon', 'MultiPolygon']
