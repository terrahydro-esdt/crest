from pathlib import Path
from fsspec.mapping import FSMap
import s3fs


class S3Path(FSMap):
    """ Wrap FSMap to mirror the pathlib.Path API """

    def __init__(self, path: "Path | str | FSMap"):
        if not isinstance(path, FSMap):
            if str(path)[:5] != 's3://':
                path = f's3://{Path(path).as_posix()}'
            path = s3fs.S3FileSystem().get_mapper(str(path))
        self.mapper = path

    def __repr__(self) -> str:
        return f'S3Path({self.as_posix()})'
    
    def __str__(self) -> str:
        return self.as_posix()
    
    def __hash__(self) -> int:
        """ Equal paths map to the same hash """
        return hash(self.path)

    def __eq__(self, other: 'S3Path') -> bool:
        """ S3Paths are equal if they reference the same path """
        return self.path == other.path

    def __getattr__(self, attr):
        """ All other attributes are pulled from the underlying FSMap """
        return getattr(self.mapper, attr)

    def __reduce__(self):
        """ Ensure S3Path object is pickled rather than just the FSMap """
        return type(self), (self.mapper,)
    
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

    @property
    def parent(self) -> 'S3Path':
        """ Same as pathlib.Path.parent """
        parent = self.path.parent.as_posix()
        mapper = self.fs.get_mapper(f's3://{parent}')
        return S3Path(mapper)
        
    def as_posix(self) -> str:
        """ Same as pathlib.Path.as_posix() """
        return f's3://{self.path.as_posix()}'

    def exists(self) -> bool:
        """ Check whether the remote S3 location exists """
        return self.fs.exists(self.mapper.root)

    def joinpath(self, *paths: "Path | str") -> 'S3Path':
        """ Append to the current S3 path """
        joined = self.path.joinpath(*paths).as_posix()
        mapper = self.fs.get_mapper(f's3://{joined}')
        return S3Path(mapper)

    def delete(self, recursive=True):
        """ Delete the file/directory from S3 """
        self.fs.rm(self.as_posix(), recursive=recursive)

    def download(self, path: Path | str, recursive=False, **kwargs):
        """ Download from S3 to a local path. Mirrors S3Filesystem.download """ 
        src = self.as_posix()
        dst = Path(path).as_posix()
        return self.fs.download(src, dst, recursive=recursive, **kwargs)