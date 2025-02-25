from collections.abc import Collection
from functools import cached_property
import numpy as np
from typing import Callable, Union

from .utils import full_resolutions


class Grid:
    def __init__(self, 
        coordinates : Union[np.ndarray, Callable],
        resolutions : Union[np.ndarray, Callable],
        table : Union[None, np.ndarray, Callable] = None,
        index : Union[None, int, list[int]] = None,
        ngrid : Union[None, int] = None,
        name  : Union[None, str] = None,
        dims  : Union[None, list[str]] = None,
    ):
        self.index = list(np.atleast_1d(index))
        self._C = coordinates
        self._R = resolutions
        self._T = np.arange(len(self))[:, None] if table is None else table
        self.name  = name or '+'.join([f'Grid_{i}' for i in self.index])
        self.dims  = dims or [f'Dim_{i}' for i in range(self.ndim)]
        self.ngrid = ngrid if ngrid is not None else len(self.index)


    def __repr__(self) -> str:
        return self.name


    def __len__(self) -> int:
        """ Length is the number of coordinate points """
        return len(self.coordinates)


    def __eq__(self, other: 'Grid') -> bool:
        """ Two grids are equal if their coordinates and resolutions match """
        # NaN != NaN, so we add in conditional to check whether both are NaN
        eq = lambda v1, v2: ((v1 == v2) | (np.isnan(v1) & np.isnan(v2))).all()
        return ( (len(self) == len(other)) and 
                 eq(self.coordinates, other.coordinates) and
                 eq(self.resolutions, other.resolutions) )


    @property
    def ndim(self) -> int:
        """ Number of dimensions """
        return self.coordinates.shape[-1]


    @property
    def anisotropic(self) -> bool:
        """ Whether resolutions are anisotropic """
        return (self.resolutions.ndim > 1) and (self.resolutions.shape[1] > 1)


    @cached_property
    def coordinates(self) -> np.ndarray:
        """ Initialization coordinates can be callable for lazy evaluation """
        return self._C() if callable(self._C) else self._C


    @cached_property
    def resolutions(self) -> np.ndarray:
        """ Initialization resolutions can be callable for lazy evaluation """
        R = np.atleast_1d(self._R() if callable(self._R) else self._R)
        return full_resolutions(self.coordinates, R)
    

    @cached_property
    def table(self) -> np.ndarray:
        """ Initialization table can be callable for lazy evaluation """
        return self._T() if callable(self._T) else self._T


    def clone(self, **kwargs) -> 'Grid':
        """ Copy the current object, modifying any given parameters """
        return Grid(**{
            'coordinates' : self.coordinates,
            'resolutions' : self.resolutions, 
            'table' : self.table,
            'index' : self.index,
            'ngrid' : self.ngrid,
            'name'  : self.name,
            'dims'  : self.dims,
        } | kwargs)


    def split(self) -> list:
        """ Split the current object into a list of composite subgrids """
        split  = lambda val: np.split(np.array(val), self.ngrid, axis=-1)
        splits = {
            'coordinates' : split(self.coordinates),
            'resolutions' : split(self.resolutions),
            'dims'        : map(list, split(self.dims)),
        }
        clone = lambda vals: self.clone(ngrid=1, **dict(zip(splits, vals)))
        return list(map(clone, zip(*splits.values())))


    @classmethod
    def combine(cls, grids: Collection['Grid'], **kwargs):
        """ Combine a list of Grid objects into one """
        return Grid(**{
            'coordinates' : lambda: np.concatenate([g.coordinates for g in grids], axis=-1),
            'resolutions' : lambda: np.concatenate([g.resolutions for g in grids], axis=-1),
            'table'       : lambda: np.concatenate([g.table for g in grids], axis=-1),
            'index'       : sum([g.index for g in grids], []),
            'ngrid'       : sum([g.ngrid for g in grids]),
            'dims'        : sum([g.dims  for g in grids], []),
            'name'        : '+'.join(g.name for g in grids),
        } | kwargs)


    def tiled_align(self, other: 'Grid') -> 'Grid':
        """ Tile this grid to match the other, if necessary """
        n_self  = self.ngrid
        n_other = other.ngrid
        if n_other > 1:

            #  First case: [a b] x3 -> [a b a b a b]
            # Second case: [a b] x3 -> [a a a b b b]
            case1 = lambda arr: np.tile(arr, n_other)
            split = lambda arr: map(case1, np.split(arr, n_self, axis=-1))
            case2 = lambda arr: np.concatenate(list(split(arr)), axis=-1)
            
            # Use the first index to break symmetry
            tile = case1 if self.index[0] < other.index[0] else case2

            return self.clone(**{
                'coordinates' : tile(self.coordinates),
                'resolutions' : tile(self.resolutions),
                'dims'        : list(tile(np.array(self.dims))),
            })
        return self


    def reorder(self, col_order: list[int]) -> (np.ndarray, 'Grid'):
        """ Create a new grid with reordered columns,
            ensuring lexicographic ordering of rows """
        row_order = np.lexsort(self.coordinates[..., col_order].T[::-1])
        return row_order, self.clone(**{
            'coordinates' : self.coordinates[..., col_order][row_order],
            'resolutions' : self.resolutions[..., col_order][:, row_order]
                if self.anisotropic else self.resolutions[..., col_order],
        })