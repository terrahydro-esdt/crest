from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

# Allow imports directly from inner crest module; e.g. rather than requiring
#   `from crest.crest.data import *`, allow `from crest.data import *`
# Note that this fully replaces the parent crest module, which also allows
#   relative imports from within crest to resolve correctly
crest_path = Path(__file__).parent.joinpath('crest')
crest_init = crest_path.joinpath('__init__.py')
crest_spec = spec_from_file_location('crest', crest_init, submodule_search_locations=[crest_path.as_posix()])
crest_modl = sys.modules['crest'] = module_from_spec(crest_spec)
crest_spec.loader.exec_module(crest_modl)

# Add tests to the crest submodules; tests should eventually be migrated into crest/crest/tests
tests_init = Path(__file__).parent.joinpath('tests', '__init__.py') 
sys.modules['crest.tests'] = module_from_spec( spec_from_file_location('tests', tests_init) )

