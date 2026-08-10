import warnings

warnings.warn(
    '"crest.archiver.Indexer" has moved to "crest.data.archiver.Indexer". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiver.Indexer import *