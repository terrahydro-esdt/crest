""" 
Defines the pytest.mark.speed decorator, allowing the use of --speed
as a command line argument when running pytest. This allows certain 
tests to be marked as 'speed', meaning they will only be run if the
--speed flag is explicitly passed when running pytest:

@pytest.mark.speed
def test_example(): 
    # This test is only run when calling pytest with --speed
    hardware_dependent_test()

The --speed flag is meant to signal that a given test is measuring
execution speed in some manner, which means that it may fail if it's
running on hardware that is not equipped to properly benchmark.

"""
import pytest


def pytest_addoption(parser):
    parser.addoption('--speed', action='store_true', default=False,
                     help='Include tests that are marked as benchmarking speed')

    parser.addoption('--examples', action="store_true", default=False,
                     help="run example tests"
                     )


def pytest_configure(config):
    config.addinivalue_line('markers', 'speed: mark test as skip by default')
    config.addinivalue_line(
        'markers', "examples: mark test as example testing")


def pytest_collection_modifyitems(config, items):
    # If --speed is not given in cli, skip marked tests
    if not config.getoption('--speed') and not config.getoption('--examples'):
        speed = pytest.mark.skip(
            reason='need --speed/--examples option to run')

        for item in items:
            if 'speed' in item.keywords or 'examples' in item.keywords:
                item.add_marker(speed)
