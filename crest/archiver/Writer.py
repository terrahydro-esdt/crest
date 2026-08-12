""" Deprecated: "crest.archiver.Writer" has moved to "crest.data.archiving.Writer". This module re-exports everything from there for backwards compatibility and will be removed in a future release. Please update your imports to use "crest.data.archiving.Writer" directly. See that module for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver.Writer" has moved to "crest.data.archiving.Writer". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiving.Writer import *