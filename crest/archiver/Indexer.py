""" Deprecated: "crest.archiver.Indexer" has moved to "crest.data.archiving.Indexer". This module re-exports everything from there for backwards compatibility and will be removed in a future release. Please update your imports to use "crest.data.archiving.Indexer" directly. See that module for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver.Indexer" has moved to "crest.data.archiving.Indexer". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiving.Indexer import *