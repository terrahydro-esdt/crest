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


Also defines the skip_on_fail marker, which allows tests to be decorated
with @pytest.mark.skip_on_fail in order to skip the remaining parameters
for a given test when that test fails on a parameter setting. 

"""
import pytest


def pytest_addoption(parser):
    parser.addoption('--speed', action='store_true', default=False,
                     help='Include tests that are marked as benchmarking speed')

    parser.addoption('--examples', action='store_true', default=False,
                     help='run example tests')


def pytest_configure(config):
    config.addinivalue_line('markers', 'speed: mark test as skip by default')
    config.addinivalue_line('markers', 
        'skip_on_fail: mark test as skip remaining parameters on failure')
    config.addinivalue_line('markers', 'examples: mark test as example testing')


def pytest_collection_modifyitems(config, items):
    # If --speed is not given in cli, skip marked tests
    if not config.getoption('--speed') and not config.getoption('--examples'):
        speed = pytest.mark.skip(
            reason='need --speed/--examples option to run')

        for item in items:
            if 'speed' in item.keywords or 'examples' in item.keywords:
                item.add_marker(speed)


# skip_on_fail marker
# https://stackoverflow.com/a/52302986
def pytest_sessionstart(session):
    session.failednames = set()

def pytest_runtest_makereport(item, call):
    markers = {marker.name for marker in item.iter_markers()}
    if call.excinfo is not None and 'skip_on_fail' in markers:
        item.session.failednames.add(item.originalname)

def pytest_runtest_setup(item):
    markers = {marker.name for marker in item.iter_markers()}
    if item.originalname in item.session.failednames and 'skip_on_fail' in markers:
        pytest.skip(item.name)
