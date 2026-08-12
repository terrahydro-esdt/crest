""" Deprecated: "crest.archiver.StageWriter" has moved to "crest.data.archiver.StageWriter". This module re-exports everything from there for backwards compatibility and will be removed in a future release. Please update your imports to use "crest.data.archiver.StageWriter" directly. See that module for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver.StageWriter" has moved to "crest.data.archiving.StageWriter". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiving.StageWriter import *