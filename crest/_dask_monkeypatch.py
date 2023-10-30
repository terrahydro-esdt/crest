""" Optimized ArrayOverlapLayer materialization / graph construction """
import numpy as np 
import dask


class PatchedHighLevelGraph(dask.highlevelgraph.HighLevelGraph):
    def cull(self, keys):
        from dask.base import flatten
        from dask.layers import Blockwise

        keys_set = set(flatten(keys))

        class InGraph:
            """Thin wrapper allowing backward compatibility with Layer.cull"""

            def __contains__(_, key):
                # Verify key is in some layer of the graph
                # Note that if a Layer's `__getitem__` function materializes
                #   the layer, `key in layer` will as well. This can be
                #   avoided by explicitly creating `Layer.__contains__`
                return any(key in layer for layer in self.layers.values())

            def __len__(_):
                return sum(map(len, self.layers.values()))

            def __iter__(_):
                raise NotImplementedError("Should not iterate over this class")

        all_ext_keys = InGraph()
        ret_layers: dict = {}
        ret_key_deps: dict = {}
        for layer_name in reversed(self._toposort_layers()):
            layer = self.layers[layer_name]
            # Let's cull the layer to produce its part of `keys`.
            # Note: use .intersection rather than & because the RHS is
            # a collections.abc.Set rather than a real set, and using &
            # would take time proportional to the size of the LHS, which
            # if there is no culling can be much bigger than the RHS.
            output_keys = {k for k in keys_set if k in layer}
            if output_keys:
                culled_layer, culled_deps = layer.cull(output_keys, all_ext_keys)
                # Update `keys` with all layer's external key dependencies, which
                # are all the layer's dependencies (`culled_deps`) excluding
                # the layer's output keys.
                external_deps = set()
                for d in culled_deps.values():
                    external_deps |= d
                external_deps -= culled_layer.get_output_keys()
                keys_set |= external_deps

                # Save the culled layer and its key dependencies
                ret_layers[layer_name] = culled_layer
                if (
                    isinstance(layer, Blockwise)
                    or isinstance(layer, dask.highlevelgraph.MaterializedLayer)
                    or (layer.is_materialized() and (len(layer) == len(culled_deps)))
                ):
                    # Don't use culled_deps to update ret_key_deps
                    # unless they are "direct" key dependencies.
                    #
                    # Note that `MaterializedLayer` is "safe", because
                    # its `cull` method will return a complete dict of
                    # direct dependencies for all keys in its subgraph.
                    # See: https://github.com/dask/dask/issues/9389
                    # for performance motivation
                    ret_key_deps.update(culled_deps)

        # Converting dict_keys to a real set lets Python optimise the set
        # intersection to iterate over the smaller of the two sets.
        ret_layers_keys = set(ret_layers.keys())
        ret_dependencies = {
            layer_name: self.dependencies[layer_name] & ret_layers_keys
            for layer_name in ret_layers
        }

        return type(self)(ret_layers, ret_dependencies, ret_key_deps)

dask.highlevelgraph.HighLevelGraph = PatchedHighLevelGraph


import dask.layers
class PatchedArrayOverlapLayer(dask.layers.ArrayOverlapLayer):
    def __getitem__(self, key):
        """Fetch / materialize a single item"""
        if hasattr(self, "_cached_dict"):
            return self._cached_dict[key]

        # Raise a TypeError if the key isn't hashable
        set(key)

        # If the first value isn't a string, it's not valid
        if not hasattr(key, "__len__") or (len(key) < 1) or not isinstance(key[0], str):
            raise KeyError

        getitem_name = f"getitem-{self.token}"
        overlap_name = f"overlap-{self.token}"

        if key[0] == overlap_name:
            from dask.array.core import concatenate3
            return (
                concatenate3,
                (
                    dask.layers.concrete,
                    dask.layers._expand_keys_around_center(
                        key, self.numblocks, getitem_name, self.axes
                    ),
                ),
            )

        if key[0] == getitem_name:
            rounded = (self.name,) + tuple(round(k) for k in key[1:])
            if rounded[1:] == tuple(key[1:]):
                return rounded
            return dask.layers.fractional_slice((self.name,) + key[1:], self.axes)
        raise KeyError

    def __len__(self):
        """This could be calculated directly to remove the numpy dependency
        and speed up the calculation slightly, but have only expanded it
        to two block dimensions so far:
            size = lambda a,b: (
                (a*b) +                   # Overlap blocks; np.prod(blocks)
                ((a-2)*(b-2))*(3*3)     + # No masked values, in middle
                ((a-2)+(b-2))*(3*2) * 2 + # One masked value, both sides
                2*(2*2) * 2               # One masked for each, both sides
            )
        """
        if getattr(self, "_cached_len", None) is None:
            blocks = np.array(self.numblocks, dtype="int32")
            n_dims = len(blocks)
            depths = [np.atleast_1d(self.axes.get(i, 0)) for i in range(n_dims)]
            active = np.array(list(map(sum, depths)), dtype=bool)
            index = np.indices(blocks, dtype="float32").reshape((n_dims, -1))

            # Generate a mask that indicates which keys are valid
            keys = np.tile(index[..., None], 3).T
            keys[0] -= 0.9
            keys[2] += 0.9
            mask = np.ones_like(keys, dtype=bool)
            mask[0] = active & (keys[0] > 0)
            mask[2] = active & (keys[2] < (blocks - 1))
            self._cached_len = mask.sum(0).prod(-1).sum() + np.prod(blocks)
        return self._cached_len

dask.layers.ArrayOverlapLayer = PatchedArrayOverlapLayer