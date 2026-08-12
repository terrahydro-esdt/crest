from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Indexer:
    """ Maps real coordinate vectors to integer indices.

    Modes:
      - 'nearest' (default): nearest neighbor index (with optional tolerance)
      - 'left'/'right': bin edges behavior
      - 'strict': exact match required

    Coordinates may be ascending or descending; must be monotonic.

    """

    coords: np.ndarray
    name: str
    mode: str = 'nearest'
    tol: float|int|None = None  # float for lat/lon; int ns for datetime64[ns]


    def __post_init__(self):
        """ Perform checks and determine coord order after initialization """
        if self.coords.ndim != 1:
            raise ValueError(f'Coordinate "{self.name}" must be 1-D')
        inc = np.all(self.coords[1:] >= self.coords[:-1])
        dec = np.all(self.coords[1:] <= self.coords[:-1])
        if not (inc or dec):
            raise ValueError(f'Coordinate "{self.name}" must be monotonic '
                             '(ascending or descending)')
        self._asc = bool(inc)


    @property
    def size(self) -> int:
        """ Total size of the coordinates """
        return int(self.coords.size)


    def find(self, coords: np.ndarray) -> np.ndarray:
        """ Find the location of a set of coordinates within the schema """
        schema = self.coords[::1 if self._asc else -1]
        search = np.searchsorted(schema, coords, side='left')
        index = np.clip(search, 0, self.size)
        if self._asc:
            return index
        return self.size - index


    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """ Convert real coordinates to integer indices """
        typed = coords.astype(self.coords.dtype)
        found = self.find(typed)
        index = np.clip(found - int(self.mode!='strict'), 0, self.size - 1)

        if self.mode == 'strict':
            i = np.where(self.coords[index] != typed)[0][:5]
            if len(i):
                raise ValueError(
                    f'{self.name} strict match failed: archive=' +
                    f'{typed[i]} != schema={self.coords[index][i]}\n' +
                    f'{self.coords.dtype=} vs {coords.dtype=}: {coords[i]}')

        elif self.mode == 'nearest':
            right = np.clip(found, 0, self.size - 1)
            dif_l = np.abs(self.coords[index] - typed)
            dif_r = np.abs(self.coords[right] - typed)
            index = np.where(dif_r < dif_l, right, index)
            if self.tol is not None:
                distance = np.abs(self.coords[index] - typed)
                i = np.where(distance > self.tol)[0][:5]
                if len(i):
                    raise ValueError(
                        f'{self.name} nearest match > {self.tol}: ' +
                        f'{self.coords[index][i]} != {typed[i]}\n' +
                        f'{self.coords.dtype=} vs {coords.dtype=}: {coords[i]}')
        return index.astype(np.int64)

