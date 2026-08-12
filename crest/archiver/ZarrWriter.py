""" Deprecated: "crest.archiver.ZarrWriter" has moved to "crest.data.archiving.ZarrWriter". This module re-exports everything from there for backwards compatibility and will be removed in a future release. Please update your imports to use "crest.data.archiving.ZarrWriter" directly. See that module for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver.ZarrWriter" has moved to "crest.data.archiving.ZarrWriter". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiving.ZarrWriter import *