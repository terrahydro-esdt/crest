from __future__ import annotations

from collections import defaultdict as dd
from functools import cached_property
from itertools import product

import dask.array as da
import xarray as xr
import pandas as pd
import numpy as np
import sparse
import dask

from .Block import Block


class SparseBlock(Block):
    """Wrap sparse block storage behind the Block sampling interface.
    
    Notes
    -----
    Sparse blocks do not include overlap from neighboring blocks due to
    efficiency concerns, and the relative rarity of valid points that would
    actually be impacted. Consequently, when window_depth is nonzero, windows
    crossing block boundaries may be excluded even if they would otherwise 
    satisfy valid_percent using neighboring data.
    
    Parameters
    ----------
    *args, **kwargs
        Block constructor arguments. Datafile selects this class
        by examining the final Dask chunk metadata for sparse.COO.

    """

    @cached_property
    def data(self) -> sparse.COO:
        """ Sparse data should be cached due to necessary re-use """
        return super().data


    @cached_property
    def coords(self) -> np.ndarray:
        """ Return original sparse coordinate rows from prepared vectors """
        indices = np.isfinite(self.data).any(-1).coords
        if not indices.size:
            raise ValueError(f'No finite coords found: {self=}')
        vectors = [vector[index] for vector, index in
                   zip(self.coord_vecs, indices)]
        return np.stack(dask.compute(*vectors), axis=-1)


    @cached_property
    def valid_windows(self) -> np.ndarray:
        """ Determine indices for all valid sparse sample windows """
        # Locations with at least one stored finite source value
        points = np.isfinite(self.data).any(-1)
    
        # The valid mask defines which individual grid cells are valid
        mask   = self.valid_mask[..., 0]
        source = mask.coords[:, mask.data]
        
        if self.is_required:
            for keys, percent in self.valid_percent.items():
                if not source.shape[1]: break
        
                n_total = int(np.prod([self.window_total[k] for k in keys]))
                require = int(np.ceil(np.nextafter(percent*n_total, 0.)))
                depths  = self.window_depth
                offsets = [
                    range(-depths[dim][1], depths[dim][0] + 1)
                    if dim in keys else (0,)
                    for dim in self.dims
                ]
                shifts = np.array(list(product(*offsets)), dtype=np.intp).T
                view   = source[:, :, None] + shifts[:, None, :]
                view   = view.reshape(self.ndim, -1)
        
                inside = ((view >= 0) &
                          (view < np.array(points.shape)[:, None])).all(0)
                unique, counts = np.unique(view[:, inside], **{
                    'axis'          : 1,
                    'return_counts' : True,
                })
                source = unique[:, counts >= require]
        
            lookup  = np.ravel_multi_index(source, points.shape)
            current = np.ravel_multi_index(points.coords, points.shape)
            valid   = np.isin(current, lookup, assume_unique=True)
        else:
            valid = np.ones(points.nnz, dtype=bool)
    
        # Remove centers whose full requested window is outside the block
        depths  = np.array(list(self.window_depth.values()))
        centers = points.coords.T
        lower   = centers - depths[:, 0] < 0
        upper   = centers + depths[:, 1] >= points.shape
        valid  &= ~(lower | upper).any(1)
    
        # Feature-bin masks also restrict which locations may be centers
        if hasattr(self, '_feature_mask'):
            mask = self._feature_mask[..., 0]
            assert np.prod(mask.numblocks) == 1, mask.numblocks

            item = mask.to_delayed(optimize_graph=False).item()
            keep = item[tuple(points.coords)].compute()

            if isinstance(keep, sparse.COO):
                keep = keep.todense()
            valid &= keep
    
        # Match Block.valid_windows: always return integer center coordinates
        largest = max(self.shape, default=0) - 1
        dtype   = np.min_scalar_type(max(0, largest))
        if dtype.itemsize < 2: dtype = np.dtype('uint16')
        return np.array(points.coords[:, valid], dtype=dtype, order='F')


    @property
    def valid_resolution(self) -> np.ndarray:
        """ Sparse matching requires uniform resolutions """
        assert(self.is_uniform), f'{self.resolution=}'
        return np.array(self.resolution)


    def _gather_windows(self,
        windows     : tuple,
        features    : list[int],
        coordinates : list[int],
    ) -> tuple:
        """Gather paired COO indices and materialize only requested outputs.

        Parameters
        ----------
        windows : tuple
            Broadcastable integer indices for the requested windows.
        features : list[int]
            Data-feature positions to read.
        coordinates : list[int]
            Coordinate-component positions to read.

        Returns
        -------
        tuple
            Dense data and optional coordinate windows, with components on
            the last axis. Temporary indices scale with the output size.
        """
        shape = np.broadcast_shapes(*(index.shape for index in windows))
        count = int(np.prod(shape))
        coord = None
        flat  = tuple(np.broadcast_to(index, shape).ravel()
                      for index in windows)

        # Index the original COO grid directly. A masked nonzero offset is
        # never reused as an offset into a different feature's data vector.
        if features:
            source = self.data
            if len(features) == source.shape[-1]:
                # All-feature requests need only one lookup through the COO
                data = source[flat].todense().reshape(shape + (len(features),))
            else:
                data = []
                for feature in features:
                    index = flat + (np.full(count, feature, dtype=np.intp),)
                    data.append(source[index].todense().reshape(shape))
                data = np.stack(data, axis=-1)
        else: data = np.empty(shape + (0,), dtype=self._coords.dtype)

        # Coordinate vectors avoid materializing any logical coordinate grid
        if coordinates:
            vectors = [self.coord_vecs[axis][flat[axis]]
                       for axis in coordinates]
            coord = np.stack(dask.compute(*vectors), axis=-1)
            coord = coord.reshape(shape + (len(coordinates),))
        return data, coord


    def _summary(self,
        data      : xr.Dataset,
        stat_keys : list[str],
        perc_keys : list[str],
        quantiles : list[int],
    ) -> xr.Dataset:
        """ Calculate sparse statistics from stored values """
        import dask.dataframe as df

        def calculate(values: xr.DataArray) -> xr.DataArray:
            """ Apply reductions to stored values of one sparse feature """
            blocks = values.data.to_delayed().ravel()
            arrays = [dask.delayed(np.append)(block.data, [np.nan])
                      for block in blocks]
            series = [dask.delayed(pd.Series)(array) for array in arrays]
            length = [1] * len(series)
            arrays = df.from_delayed(series).to_dask_array(lengths=length)
            extra  = dd(int, {
                'null' : lambda array: values.size - da.sum(da.isfinite(array)),
            })
            coord = xr.Variable('statistics', stat_keys)
            nan_f = lambda key: getattr(da, f'nan{key}', extra[key])
            value = lambda key: nan_f(key)(arrays).astype('float32')
            toset = lambda key: xr.DataArray(value(key))
            stats = xr.concat(map(toset, stat_keys), coord)
            stats = stats.to_dataset('statistics')
            stats[perc_keys] = da.percentile(arrays, quantiles, **{
                'internal_method' : 'tdigest',
            })
            return stats.to_array('statistics')
        return data.apply(calculate)


    def _get_coordinate(self, index: int) -> da.Array:
        """ Return sparse coordinate values at stored source locations """
        points = np.isfinite(self.data).any(-1)
        coords = points.coords
    
        values = dask.compute(self.coord_vecs[index][coords[index]])[0]
        array  = sparse.COO(coords, values, shape=points.shape,
                            fill_value=np.nan)
    
        return da.from_array(array[..., None], **{
            'chunks'  : self._data[0].chunks,
            'asarray' : False,
            'name'    : False,
        })