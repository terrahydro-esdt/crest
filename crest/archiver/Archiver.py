""" Deprecated: "crest.archiver.Archiver" has moved to "crest.data.archiver.Archiver". This module re-exports everything from there for backwards compatibility and will be removed in a future release. Please update your imports to use "crest.data.archiver.Archiver" directly. See that module for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver.Archiver" has moved to "crest.data.archiving.Archiver". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiving.Archiver import *