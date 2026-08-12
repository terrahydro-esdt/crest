""" Deprecated: "crest.archiver.Writer" has moved to "crest.data.archiver.Writer". This module re-exports everything from there for backwards compatibility and will be removed in a future release. Please update your imports to use "crest.data.archiver.Writer" directly. See that module for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver.Writer" has moved to "crest.data.archiver.Writer". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiver.Writer import *