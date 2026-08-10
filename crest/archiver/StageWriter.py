import warnings

warnings.warn(
    '"crest.archiver.StageWriter" has moved to "crest.data.archiver.StageWriter". '
    'Please update your imports.',
    DeprecationWarning,
    stacklevel=2,
)

from crest.data.archiver.StageWriter import *