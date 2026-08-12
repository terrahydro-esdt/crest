""" Deprecated: "crest.archiver" has moved to "crest.data.archiving". This
package re-exports everything from there for backwards compatibility and
will be removed in a future release. Please update your imports to use
"crest.data.archiver" directly. See that package for the real documentation.
"""

import warnings

warnings.warn(
    '"crest.archiver" has moved to "crest.data.archiving". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiving import *