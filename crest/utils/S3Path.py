from pathlib import Path
from fsspec.mapping import FSMap
import s3fs


class S3Path(FSMap):
    """ Wrap FSMap to mirror the pathlib.Path API """

    def __init__(self, path: "Path | str | FSMap"):
        if not isinstance(path, FSMap):
            path = s3fs.S3FileSystem().get_mapper(path)
        self.mapper = path

    def __repr__(self) -> str:
        return f'S3Path({self.as_posix()})'
        
    def __hash__(self) -> int:
        """ Equal paths map to the same hash """
        return hash(self.path)

    def __eq__(self, other: 'S3Path') -> bool:
        """ S3Paths are equal if they reference the same path """
        return self.path == other.path

    def __getattr__(self, attr):
        """ All other attributes are pulled from the underlying FSMap """
        return getattr(self.mapper, attr)
        
    @property
    def path(self) -> Path:
        """ Return a Path object representing the S3 path """
        return Path(self.mapper.root)
        
    @property
    def stem(self) -> str:
        """ Same as pathlib.Path.stem """
        return self.path.stem

    @property
    def name(self) -> str:
        """ Same as pathlib.Path.name """
        return self.path.name
        
    def as_posix(self) -> str:
        """ Same as pathlib.Path.as_posix() """
        return self.path.as_posix()

    def exists(self) -> bool:
        """ Check whether the remote S3 location exists """
        return self.mapper.fs.exists(self.mapper.root)

    def joinpath(self, *paths: "Path | str") -> 'S3Path':
        """ Append to the current S3 path """
        joined = self.path.joinpath(*paths)
        mapper = self.mapper.fs.get_mapper(joined)
        return S3Path(mapper)