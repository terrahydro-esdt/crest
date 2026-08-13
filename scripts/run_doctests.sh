#!/usr/bin/env bash
# Run all module docstring (Examples) doctests found under crest/, via pytest.
#
# Usage:
#     scripts/run_doctests.sh

set -euo pipefail
cd "$(dirname "$0")/.."

files=$(grep -rl '>>>' crest --include='*.py')
pytest --color=yes --doctest-modules --import-mode=importlib $files
