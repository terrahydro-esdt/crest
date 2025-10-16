#!/usr/bin/env python
"""
Run doctests for a specified module.

Usage:
    python run_doctest.py crest.model.HierarchalTensorGraph
    python run_doctest.py crest.model.HierarchalTensorGraph -v

Note: This script is a workaround because 
    python -m doctest 
only works with file paths, not module paths, and it doesn't handle 
relative imports well which is how CREST is setup.

"""
import sys
import doctest

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_doctest.py <module_name> [-v]")
        sys.exit(1)

    module_name = sys.argv[1]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    # Import the module
    mod = __import__(module_name, fromlist=[''])

    # Get the actual module (not a class)
    if module_name in sys.modules:
        mod = sys.modules[module_name]

    # Run doctest
    result = doctest.testmod(mod, verbose=verbose)

    print(f'\n=== Doctest Summary ===')
    print(f'Tests attempted: {result.attempted}')
    print(f'Tests passed: {result.attempted - result.failed}')
    print(f'Tests failed: {result.failed}')

    sys.exit(0 if result.failed == 0 else 1)
