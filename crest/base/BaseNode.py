from __future__ import annotations

from collections.abc import Callable, Collection
from sklearn.metrics import r2_score
from scipy.stats import linregress
from functools import cached_property, partial, reduce
from operator import and_
from abc import abstractmethod

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import logging
import hashlib
import keras
import jax
import tlz
import re

from crest.utils import classproperty, plot_to_array
from crest.base import BaseAbstract

logger = logging.getLogger(__name__)


# BaseNode.inputs/outputs type annotation. These two dictionaries should
# follow the format of:
# { '<SOURCE>' :         # String indicating where features are found
#   { '<FEATURE>' :      # String indicating features that are needed
#     { '<COORDINATE>' : # String indicating coordinate dimensions; either:
#       total            # - int, giving total size of windows along this dim
#     | None             # - None, indicating windows can be any size
#     | (left, right)    # - Collection of two ints, defining left/right window
#                        #   extent (total=left+right+1)
# } } }
#
# _coord_type defines the inner coordinate dictionary annotation
COORD_TYPE = dict[str, Collection[int] | int | None]

# IO_TYPE defines the full inputs/outputs annotation
IO_TYPE = dict[str, dict[str, COORD_TYPE]]

# Handle all Keras backend types
try:
    GenericSpec = keras.InputSpec | tf.TensorSpec
    GenericTensor = keras.KerasTensor | tf.Tensor | jax.Array | np.ndarray
except TypeError:
    # This fallback only affects doc builds, not real type checking.
    GenericSpec = object
    GenericTensor = object

class BaseNode(BaseAbstract):
    """BaseNode provides template for machine learning models to inherit from.

    Notes
    -----
    Any Model(BaseNode) definition must define Model.inputs and Model.outputs,
    i.e. class-level dictionary attributes. These dictionaries define the input
    and output shapes for features coming into, and going out of, the model.
    BaseNode objects will then automatically have inputs_spec and outputs_spec
    attributes defined, allowing the specs to be directly passed into any
    HierarchalTensorGraph definition. As well, if forward and inverse
    normalization functions are given to the Model upon initialization, pre-
    and post-processing nodes will be added to the Model, which will normalize
    and de-normalize the inputs and outputs of the Model, respectively.

    Parameters
    ----------
    *normalize : Transform | tuple[Callable, Callable]
        A pair of functions defining (forward transform, inverse transform)
        for the input and output features, respectively. By default, no
        pre- or post-processing is applied. A Transform object can also be
        given directly, in which case transformations are applied by feature,
        as defined in the given Transform.
    loss       : str | Callable | dict[str, str | Callable | dict]
        Loss function that should be used for this Model. This can be passed
        as a standard keras loss string (e.g. 'mse'); as an arbitrary callable
        which has the signature loss(y_true, y_pred) -> float; or, if multiple
        outputs are returned from this Model, a dictionary mapping output
        feature names to the respective loss str or callable for that feature.
    transform_loss : bool | tuple[Callable]
        Determines whether losses should be applied before or after the post-
        processing (if any exists). In other words, with transform_loss=True
        (the default), losses will be calculated on the transformed outputs
        of the model (prior to post-processing), rather than the inverse
        transformed version. This can help avoid needing to weight model losses
        differently, when multiple sub-model losses are being used to train a
        larger hierarchal model. Note this has no effect if no ``*normalize``
        functions are given.
        Alternatively, a tuple of callables (in the same format as normalize)
        can be given to use as a different normalization procedure than the
        feature normalization.
    debug          : bool
        Determines whether y_true and y_pred statistics should be printed on
        each batch inside the loss wrapper (default: False). Note this has no
        effect if ``*normalize`` is not given or transform_loss=False.
    plot_scatter   : bool
        Determines whether scatter plots should be created in the loss function
        to add to TensorBoard. Setting to True will add target vs predicted
        scatter plots for each loss function update, which in turn slows down
        the training process.

    """
    inputs  : IO_TYPE
    outputs : IO_TYPE

    # Setting this to True in an inheriting class will automatically add
    # distance-based features to the input dictionary given to the model
    # call function; e.g. a 'distance:ERA5.zarr:SMAP.zarr' feature which
    # contains the dict {'latitude': latitude_distance, ...} for all dims
    include_distances = False


    def __init__(self,
        normalize      = (),
        loss           : str | Callable | dict[str, str|Callable|dict] = 'mse',
        transform_loss : bool | tuple[Callable, Callable] = True,
        debug          : bool = False,
        plot_scatter   : bool = False,
        plot_histogram : bool = False,
        use_raw_pred   : bool = False,
    ):
        self.normalize      = getattr(normalize, 'by_feature', normalize)
        self.loss           = loss
        self.debug          = debug
        self.transform_loss = transform_loss
        self.plot_scatter   = plot_scatter
        self.plot_histogram = plot_histogram
        self.use_raw_pred   = use_raw_pred


    @property
    def __name__(self) -> str:
        return self.__class__.__name__


    @abstractmethod
    def call(self,
        X        : dict[str, GenericTensor],
        training : bool,
    ) -> dict[str, GenericTensor]:
        """ Inheriting classes must define a `call` method, which takes
            as input a dictionary of {feature name: Input Tensor}, and
            returns a dictionary of {output feature: Output Tensor}. """
        raise NotImplementedError(f'{self}.call() must be implemented')


    @cached_property
    def node(self) -> 'crest.model.Node.Node':  # noqa: UP037, F821
        from crest.model import Node
        return Node(**{
            'node': _NodeWrap(self, f'{self}-call', self.plot_histogram),
            'name': f'{self}-Node',
            'inputs': self.inputs_spec,
            'outputs': self.outputs_spec,
        })

    @cached_property
    def graph(self) -> 'HierarchalTensorGraph':  # noqa: UP037, F821
        from crest.model import HierarchalTensorGraph as HTG
        """ Create the default base node graph object """
        htg = HTG(f'{self}')
        htg.add_edge('input',self.node)
        htg.add_edge(self.node,'output')
        return htg

    @property
    def losses(self) -> dict[str, Callable]:
        """ Return a dictionary of losses, one per output feature """
        get_loss = lambda loss: (loss if not isinstance(loss, (str, dict))
                                      else keras.losses.get(loss))

        # Allow loss specification via config dict (see keras.losses.get)
        if isinstance(self.loss, dict) and ('class_name' not in self.loss):
            losses = {k: get_loss(loss) for k, loss in self.loss.items()}
        else:
            losses = {k: get_loss(self.loss) for k in self.output_spec}

        raw_preds = (lambda:self._raw_model_out) if self.use_raw_pred else None
        wrap_loss = partial(loss_wrapper, **{
            'scope'        : f'{self}',
            'transform'    : getattr(self, 'loss_transformer', None),
            'postprocess'  : getattr(self, 'postprocess', None),
            'plot_scatter' : self.plot_scatter,
            'debug'        : self.debug,
            'y_pred_raw'   : raw_preds,
        })

        # Create a wrapper for loss functions to enable masking NaN
        # targets/predictions, and applying different normalization
        return dict(zip(losses, map(wrap_loss, losses, losses.values())))


    @classproperty
    def inputs_spec(cls) -> dict[str, GenericSpec]:
        """ Convert input shape dictionary into TensorSpec objects """
        return cls._generate_spec(cls.inputs, 'inp')
    @classproperty
    def input_spec(cls) -> dict[str, GenericSpec]:
        """ Backward compatibility """
        return cls.inputs_spec

    @classproperty
    def outputs_spec(cls) -> dict[str, GenericSpec]:
        """ Convert output shape dictionary into TensorSpec objects """
        return cls._generate_spec(cls.outputs, 'out')
    @classproperty
    def output_spec(cls) -> dict[str, GenericSpec]:
        """ Backward compatibility """
        return cls.outputs_spec

    @classmethod
    def _generate_spec(cls,
        io_shapes_spec : IO_TYPE,
        suffix         : str,
    ) -> dict[str, keras.InputSpec]:
        """ Generate a TensorSpec dict from the given input/output config """

        _valid_first = re.compile(r"^[A-Za-z0-9.]")
        _valid_rest  = re.compile(r"[^A-Za-z0-9_.>-]")

        def safe_name(k: str) -> str:
            s = _valid_rest.sub("_", k.strip())
            if not _valid_first.match(s):
                s = "k_" + s
            # ensure uniqueness even if two keys sanitize the same
            h = hashlib.md5(k.encode("utf-8")).hexdigest()[:5]
            return f"{s}_{h}_{suffix}"

        def parse(coord_shapes: COORD_TYPE) -> list[int | None]:
            """ Convert window extent tuple to total shape """
            if not len(coord_shapes): return [None]
            # Handle left/right extent being used, and order by coordinate key
            get_total = lambda s: sum(s)+1 if isinstance(s, Collection) else s
            _,ordered = zip(*sorted(coord_shapes.items(), key=lambda kv:kv[0]))
            return [None] + list(map(get_total, ordered))

        specs = {feature: keras.InputSpec(**{
                    'shape' : parse(coord_shapes),
                    'name'  : safe_name(feature),
                    'dtype' : 'float32',
                })
                for source, feature_shapes in io_shapes_spec.items()
                for feature,  coord_shapes in feature_shapes.items()}

        if suffix == 'inp' and getattr(cls, 'include_distances', False):
            for s1, features in cls.inputs.items():
                inp_shape = dict(next(iter(features.values()))) | {'z': 3}

                for s2 in cls.outputs:
                    name = f'distance:{s1}:{s2}'
                    specs[name] = keras.InputSpec(**{
                        'shape' : parse(inp_shape),
                        'name'  : safe_name(name),
                        'dtype' : 'float32',
                    })
        return specs


    def _add_data_transform(self):
        from crest.model import Node
        """ Add pre-/post-processing functions to the graph """
        assert(getattr(self, 'preprocess', None) is None), \
            f'Should only call {self}._add_data_transform once'

        # Define the pre- and postprocessing transformers
        self.preprocess, self.postprocess = self.normalize

        # Set the loss transformer if it was requested to be used
        if self.transform_loss is True:
            self.loss_transformer = self.preprocess

        # Otherwise, if separate forward/inverse loss transformers were given,
        # these should be used for loss transformation and model postprocessing
        elif self.transform_loss:
            self.loss_transformer, self.postprocess = self.transform_loss

        wrapper = partial(_NodeWrap, plot_histogram=self.plot_histogram)
        preprocess = wrapper(self.preprocess,  f'{self}-preprocess')
        postprocess = wrapper(self.postprocess, f'{self}-postprocess')

        # Instantiate graphs for each node to define input/output specs
        preprocess = Node(preprocess, self.node.inputs,
                          self.node.inputs, 'preprocess')
        postprocess = Node(postprocess, self.node.outputs,
                           self.node.outputs, 'postprocess')

        # Add new edges for pre-/post-processing
        self.graph.remove_edge('input',self.node)
        self.graph.remove_edge(self.node,'output')
        self.graph.add_edges_from([
            ('input', preprocess),
            (preprocess, self.node),
            (self.node, postprocess),
            (postprocess, 'output'),
        ])


    def _call(self, X: dict[str, GenericTensor | list], training=False):
        """ Wraps self.call to provide data parsing / shape validation """
        def unstack(v):
            if isinstance(v, list):
                return v
            return [v]
            if v.shape[0] is not None:
                return tf.unstack(v)
            return [v]

        # Group features by source
        Z = getattr(X, 'copy', lambda: X)()
        Y = {source: {k: Z.pop(k) for k, coords in features.items()}
                for source, features in self.inputs.items()}

        if getattr(self, 'include_distances', False):
            for k in list(Z):
                if k.startswith('distance:'):
                    _, src, tgt = k.split(':')
                    for i,v in enumerate(keras.ops.unstack(Z.pop(k), axis=-1)):
                        Y[src][f'distance:{tgt}:{i}'] = v
        X = Z | Y

        for source, features in X.items():
            if isinstance(features, dict):
                # Handle any coordinate features
                X[source] = self.convert_coords(features)

                # One-hot encode various class features
                X[source] = self.convert_onehot(features, {
                    'landcover_class_0' : (17, None),
                    'slt' : (7, None),
                    'tvl' : (20, [0, 1, 6, 8, 9, 10, 12, 15, 16, 19]),
                    'tvh' : (20, [2, 3, 4, 5, 17, 18]),
                })
            X[source] = {k: unstack(v) for k,v in features.items()}

        # Wrap output(s) with output_spec dictionary if not already
        out = self.call(X, training)
        if not isinstance(out, (dict, list, tuple)):
            out = [out]
        if isinstance(out, (list, tuple)):
            if len(out) != len(self.output_spec):
                raise ValueError(f'Expected {len(self.output_spec)} outputs'
                        + f' from {self}, but received {len(out)} value(s)')
            out = dict(zip(self.output_spec.keys(), out))

        def match_shape(feature):
            """ Match output to spec shape by adding any necessary dims """
            matching = output = out[feature]
            expected = self.output_spec[feature].shape

            # If all dims accounted for, just perform a verification
            current = keras.ops.shape(matching)
            if len(current) == len(expected):
                for c, e in zip(current, expected):
                    if (e is None) or not isinstance(c, int):
                        continue
                    if c != e:
                        err = f'{feature} output shape {expected=}: {output}'
                        raise ValueError(err)
                return matching

            # Remove all empty dimensions
            empty = [i for i, shape in enumerate(current)
                     if isinstance(shape, int) and shape == 1]

            # Only dims which are not None can be removed
            for i in list(empty):
                if len(expected) > i and expected[i] is None:
                    empty.remove(i)
            matching = keras.ops.squeeze(matching, empty)

            # Verify all dimension shapes match the respective spec shape
            for i, shape in enumerate(expected):
                if shape is None:
                    continue
                current = keras.ops.shape(matching)

                # Add empty dim at the end (e.g. [2, 2] -> [2, 2, 1])
                if len(current) <= i:
                    matching = keras.ops.expand_dims(matching, i)
                    current = keras.ops.shape(matching)

                # Only validate dims with known shapes
                if isinstance(current[i], int):
                    if shape not in [1, current[i]]:
                        err = f'{feature} output shape {expected=}: {output}'
                        raise ValueError(err)

                    # Add empty dim prior to others (e.g. [2, 2] -> [2, 1, 2])
                    if current[i] != shape:
                        matching = keras.ops.expand_dims(matching, i)
            return matching

        # Ensure output shapes match their output spec shape
        return dict(zip(out, map(match_shape, out)))


    def __call__(self, X):
        """ Calling a BaseNode passes input through its graph """
        return self.graph( getattr(X, 'copy', lambda: X)() )


    def __init_subclass__(cls, *args, **kwargs):
        """ Verifies child classes defined inputs/outputs dictionaries """
        super().__init_subclass__()
        for key in ['inputs', 'outputs']:
            cls.verify_type(getattr(cls, key, None), IO_TYPE, f'{cls}.{key}')


    def __post_init__(self):
        """ Called after object initialization to ensure all
            keras/tensorflow objects are already added to this object """
        super().__post_init__()

        # If normalization functions were given, add pre-/post-processing
        if len(self.normalize):
            count = len(self.normalize) == 2
            funcs = all(map(callable, self.normalize))
            assert(count and funcs), f'Arguments to {self} should be two ' + \
                'callables which define forward and inverse transformations'
            self._add_data_transform()

        # Set the loss transformer if only it was given
        elif not isinstance(self.transform_loss, bool):

            # Allow just a single transformer to be given (rather than
            # requiring both forward/inverse, since inverse is not used)
            if isinstance(self.transform_loss, (list, tuple)):
                self.loss_transformer = self.transform_loss[0]
            else: self.loss_transformer = self.transform_loss
            assert(callable(self.loss_transformer))

        # self.normalize needs to be set if transform_loss = True
        elif self.transform_loss:
            raise ValueError('Cannot transform loss if no functions are given')

        # Verify output features can actually be transformed, since
        # the loss transformer will have no effect otherwise
        if hasattr(self, 'loss_transformer'):
            unable = [f for f in self.output_spec if not getattr(
                self.loss_transformer, 'can_transform', lambda f: True)(f)]

            if unable:
                lt = self.loss_transformer
                error = f'{self} requested loss transformation, but '
                error+= f'{getattr(lt, "__name__", lt)} cannot transform'
                solve = f'Ensure {unable} are in the stats DataArray used to '
                solve+= 'initialize Transformer; or, set transform_loss=False'

                # If no outputs can be transformed, raise an exception
                if len(unable) == len(self.output_spec):
                    raise ValueError(f'{error} any of the outputs!\n{solve}')

                # Otherwise just log a warning
                logger.warning(f'{error} some outputs: {unable}\n{solve}')

    def convert_onehot(self, X: dict, onehot_classes: dict) -> dict:
        """ One-hot encode the given class features.

        Parameters
        ----------
        X : dict
            Data source dictionary like {feature_name : value_tensor}. Values
            should be converted to the appropriate dtype before being passed
            to this function, and will be cast to int32 if not already an
            integer dtype.
        onehot_classes : dict
            Dictionary like {feature_name : (total classes, include classes)}.
            This allows certain classes to be excluded from the returned dict
            of features if desired, e.g. if a class is too rare to be useful.
            Note that `include classes` is 0-based, so e.g. total classes is
            7, then include classes must be in the range 0 to 6. Also note that
            `None` can be given for `include classes` in order to include all.

        Returns
        -------
        dict
            The original data source dictionary, minus any features that were
            contained in onehot_classes, plus the new one-hot encoded features
            for any that were available (formatted 'feature_classnum').

        """
        for feature, (total_cls, include_cls) in onehot_classes.items():
            if feature in X:
                values = X.pop(feature)
                if not values.dtype.is_integer:
                    values = keras.ops.cast(values, 'int32')
                onehot = keras.ops.one_hot(values, total_cls)
                for i in include_cls or range(total_cls):
                    X[f'{feature}_{i}'] = onehot[..., i]
        return X


    def convert_coords(self, X: dict) -> dict:
        """ Reshape coordinates to be stackable alongside other features """
        coord_names = ['datetime', 'latitude', 'longitude']
        first_coord = tlz.first(tlz.dissoc(X, *coord_names).values())
        coord_shape = keras.ops.shape(first_coord[0]
                        if isinstance(first_coord, list) else first_coord)
        format_list = lambda v: [v] if isinstance(first_coord, list) else v

        # Expected shape of other features is [batch, datetime, lat, lon]. If
        # we don't find a tensor with four dimensions, just return the original
        # dict. Better handling for other tensor shapes (e.g. static) should be
        # implemented in the future.
        if len(coord_shape) != 4: return X

        def add_hist(data, days, label):
            """ Add histogram of converted datetime values for final items """
            if not self.plot_histogram: return
            def add():
                with keras.name_scope(''):
                    tf.summary.histogram(**{
                        'name' : f'{self}-call/datetime_{days}_last{label}',
                        'data' : data,
                    })
            tf.cond(tf.summary.should_record_summaries(), add, lambda: None)

        ndt,nlt,nln = coord_shape[1], coord_shape[2], coord_shape[3]
        for name in coord_names:
            if name in X:

                # Tile coordinate features into 3d cube
                coordinate = X.pop(name)
                tiles, idx = {
                    'datetime'  : ([1, 1, nlt, nln],
                                   (slice(None), slice(None), None, None)),
                    'latitude'  : ([1, ndt, 1, nln],
                                   (slice(None), None, slice(None), None)),
                    'longitude' : ([1, ndt, nlt, 1],
                                   (slice(None), None, None, slice(None))),
                }[name]

                if isinstance(coordinate, list):
                    coordinate = coordinate[0]

                # Just duplicate latitude and longitude along other dimensions
                if name != 'datetime':
                    X[name] = format_list(
                        keras.ops.tile(coordinate[idx], tiles))
                    continue

                # Convert datetime to value between [0, minutes in day & year]
                for n_days in [1, 365.2425]:
                    phase_mins = 60 * 24 * n_days
                    coord_mins = keras.ops.mod(coordinate, phase_mins)

                    # Histograms of datetime for final timestep and sample
                    add_hist(coord_mins[:,-1], n_days, 'step')
                    add_hist(coord_mins[-1], n_days, 'sample')

                    # Convert to radians
                    coord_rads = coord_mins * 2 * np.pi / phase_mins

                    # Convert datetime to set of two periodic sin/cos features
                    for op in ['sin', 'cos']:
                        periodic = getattr(keras.ops, op)(coord_rads)
                        X[f'{op}_{n_days:.0f}'] = format_list(
                            keras.ops.tile(periodic[idx], tiles))

                        # Histograms of final periodic value for step & sample
                        add_hist(periodic[:,-1], n_days, f'step_{op}')
                        add_hist(periodic[-1], n_days, f'sample_{op}')
        return X



class _NodeWrap(keras.layers.Layer):
    """ Wraps a callable / object in a keras layer, adding any internal
        tensorflow / keras objects to the wrapped object to allow keras
        to track the weights correctly.

        This can be used to add any functionality around nodes, e.g. including
        tensorboard histogram visualization for all outputs of this node. Note
        that histograms for inputs to this node may not work, as inputs coming
        into the node might not be within the tensorflow graph (e.g. they may
        be raw data inputs).

        Parameters
        ----------
        obj  : Callable | object
            The given object should either be callable itself, or have
            an obj._call method that will be used to call the object.
        name : str
            The name which should be assigned to this layer, which will
            default to the name of the passed object if nothing is passed.

    """
    def __init__(self,
        obj  : Callable,
        name : str = '',
        plot_histogram : bool = False,
    ):
        super().__init__(name=name or getattr(obj, 'name', str(obj)))

        # Add all tensorflow/keras objects to allow weight tracking
        keywords = ['tensorflow', 'keras', 'terrahydro']
        for k, v in obj.__dict__.items():
            if any(name in str(type(v)) for name in keywords):
                setattr(self, k, v)

        self.obj = obj
        self.obj_name = getattr(obj, '__name__', repr(obj))
        self.plot_histogram = plot_histogram


    def __repr__(self):
        return repr(self.obj)


    def build(self, input_shape):
        super().build(input_shape)


    @property
    def __name__(self):
        return f'_NodeWrap({self.obj.__class__.__name__})'


    def call(self, X, *args, **kwargs):
        """ Adds output histograms for tensorboard visualization """
        with keras.name_scope(''):  # noqa: SIM117
            with keras.name_scope(self.name):
                self._add_histograms(X, 'input')
                if 'args' in kwargs and 'kwargs' in kwargs:
                    args, kwargs = kwargs['args'], kwargs['kwargs']
                out = getattr(self.obj, '_call', self.obj)(X, *args, **kwargs)
                self.obj._raw_model_out = out
                self._add_histograms(out, 'output', desc=self.obj_name)

                # Add histograms for all weights
                # def add():
                #     with tf.name_scope(f'{self.obj_name}/weights'):
                #         for w in self.weights:
                #             tf.summary.histogram(w.path, w,
                #                                  description=w.name)
                # tf.cond(tf.summary.should_record_summaries(), add,
                #         lambda: None)

        return out


    def get_config(self):
        return super().get_config() | {
            'obj'      : self.obj,
            'obj_name' : self.obj_name}


    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def _add_histograms(self, X, scope: str, desc: str | None = None):
        """ Log the given X dict/tensor(s) histograms under the given scope """
        if not self.plot_histogram: return
        def add():
            hist = lambda k, v: tf.summary.histogram(**{
                'name'        : k.replace('@', '.'),
                'data'        : tf.identity(v),
                'description' : desc,
            })
            if isinstance(X, dict):
                for k, v in X.items():
                    hist(f'{k}/{scope}', v)
            else: hist(scope, X)
        tf.cond(tf.summary.should_record_summaries(), add, lambda: None)



def loss_wrapper(
    feature      : str,
    loss_func    : Callable,
    scope        : str = '',
    transform    : Callable | None = None,
    postprocess  : Callable | None = None,
    y_pred_raw   : Callable | None = None,
    plot_scatter : bool = False,
    debug        : bool = False,
) -> Callable:
    """ Wrapper for loss functions to enable NaN handling and normalization.

    Notes
    -----
    This wrapper assumes that the loss function it is wrapping will be
    applied to y_true and y_pred values that are already transformed
    back into their original domains (if any normalization took place
    during the model pipeline); i.e. y_true is the original (un-normalized)
    target value, and y_pred is the inverse-normalized model output.

    Parameters
    ----------
    feature      : str,
        Name of the feature that the loss is calculated for (i.e. the
        target feature name).
    loss_func    : Callable,
        The original loss function that should be wrapped. This function
        should have the signature `loss_func(y_true, y_pred) -> loss_scalar`.
    scope        : str
        Scope that any histograms / tensorboard items should be grouped in.
    transform    : Callable | None
        Transformation function that should be applied to the targets and
        predictions before calculating the loss.
    postprocess  : Callable | None
        Postprocessing normalization function that was _already_ applied
        to the predictions (only used for labeling).
    plot_scatter : bool
        Whether to generate scatter plots for each batch that passes through
        the loss function, and log them to tensorboard. Note that this slows
        down the function considerably, due to tensorflow needing to execute
        the matplotlib plotting outside of its compiled computation graph.
    debug        : bool
        Whether debugging logs should be printed to the terminal for each
        batch (e.g. data statistics).

    Returns
    -------
    Callable
        Returns the loss function, which takes y_true and y_pred
        as input, and returns the loss value as output.

    """

    def calculate_loss(y_true, y_pred):
        """ Mask NaNs and apply normalization before calculating loss """

        def gather_and_log(key, **kwargs):
            """ Extract, log, and return feature values and valid indicator """
            labels, values = map(list, zip(*kwargs.items()))

            for i, val in enumerate(values):
                if isinstance(val, dict):
                    values[i] = val[feature]

            flat = [keras.ops.reshape(v, (-1,)) for v in values]
            mask = reduce(and_, map(keras.ops.isfinite, flat))
            flag = keras.ops.any(mask)
            logging(f'{scope}/{feature}/{key}/', dict(zip(labels, flat)))
            return values, flag

        # Label the targets/predictions with their transformation
        ylabel = lambda y,f: (
            y if f is None else f'{getattr(f,"__name__",f)}({y})')
        labels = ['y_true', ylabel('y_pred', postprocess)]
        y_dict = dict(zip(labels, [y_true, y_pred]))
        values, flag = gather_and_log('original', **y_dict)

        # Preprocess the target if we want to calculate normalized loss
        if transform is not None:
            y_true, y_pred = values
            y_true = transform({feature: y_true})
            y_pred = (y_pred_raw or (lambda: transform({feature: y_pred})))()

            labels = [ylabel('y_true', transform), 'y_pred']
            y_dict = dict(zip(labels, [y_true, y_pred]))
            values, flag = gather_and_log('transform', **y_dict)

        # Calculate and log the final loss, then return if any values are valid
        loss = loss_func(*values)
        logging(f'{scope}/{feature}/loss/', {'loss': loss}, False)
        return keras.ops.cond(flag, lambda: keras.ops.mean(loss), lambda: 0.)

    # Accumulate batches over multiple steps for more efficient plotting
    batches = {}

    # Defines various logging functionality in a separate helper method
    def logging(scope: str, arrs: dict, do_scatter=plot_scatter) -> None:
        """ Output debug logs, scatter plots, histograms """

        def scatter(y1: np.ndarray, y2: np.ndarray, **kwargs) -> np.ndarray:
            """ Create scatter plot and return the image as a numpy array """
            scope = kwargs['key']
            if scope not in batches:
                return np.empty((0,0,0), dtype=np.uint8)
            k1 = kwargs.get('y1_label', 'y_true')
            y1 = batches[scope][k1]
            k2 = kwargs.get('y2_label', 'y_pred')
            y2 = batches[scope][k2]

            y1 = np.concatenate(y1, axis=0).flatten()
            y2 = np.concatenate(y2, axis=0).flatten()
            batches[scope].clear()

            mask = np.isfinite(y1) & np.isfinite(y2)
            y1 = y1[mask]
            y2 = y2[mask]

            # Write the plot image to the array variable
            with plot_to_array(dpi=150, width=4, height=3.8) as array:
                try:    r2 = linregress(y1, y2)[2] ** 2 # pyright: ignore
                except: r2 = np.nan
                try:    R2 = r2_score(y1, y2)
                except: R2 = np.nan

                # Scatter plot with KDE contours
                p = sns.jointplot(x=y2, y=y1)
                try:
                    p.plot_joint(sns.kdeplot, color='r', zorder=1,
                                 levels=6, alpha=0.7)
                except ValueError as e:
                    print(f'\nWarning "{e}": {scope=}' +
                          f'\n\t{k1}={y1}\n\t{k2}={y2}')

                # 1:1 diagonal line
                plt.axline((0,0), slope=1,ls='--',color='k',alpha=0.5,zorder=2)

                # Plot labels
                plt.ticklabel_format(style='sci',axis='both',scilimits=(-2,3))
                plt.ylabel(kwargs.get('y1_label', 'y_true'), fontsize=18)
                plt.xlabel(kwargs.get('y2_label', 'y_pred'), fontsize=18)
                plt.title('  '.join([
                    rf'$R^{{2}}$ = {R2:.2f}',
                    rf'$r^{{2}}$ = {r2:.2f}',
                    f'N = {len(y1)}',
                ]), fontsize=18, y=0.95)

                # Use the same extent limits for both axes
                minim = min(plt.xlim()[0], plt.ylim()[0])
                maxim = max(plt.xlim()[1], plt.ylim()[1])
                plt.xlim((minim, maxim))
                plt.ylim((minim, maxim))
                return array

        # Only tensorflow supported for logging to tensorboard
        if keras.backend.backend() != 'tensorflow':
            return

        # Remove NaNs
        mask = reduce(and_, map(tf.math.is_finite, arrs.values()))
        arrs = {k: tf.boolean_mask(v, mask) if len(mask.shape) else v
                for k,v in arrs.items()}

        # Print various statistics for debugging
        if debug:
            align = max(map(len, arrs))
            stats = lambda y: [(k, getattr(tf, k)(y)) for k in [
                'shape', 'reduce_min', 'reduce_max', 'reduce_mean']]
            tf.print(f'\n\n{scope}:', *[ output
                for k, arr in arrs.items()
                for output in [f'\n{k:>{align}}'] + stats(arr) ])

        record = tf.summary.should_record_summaries()
        # record = any(v.size > 100 for vals in batches.values()
        #              for v in vals.values())
        if record:#isinstance(record, bool) and record:
            # Log array histograms to tensorboard; Can't contain '/'?
            # with keras.name_scope(scope.replace("@",".")):
            with tf.name_scope(scope.replace("@",".")):
                for k, v in arrs.items():
                    tf.summary.histogram(k, v)

        # Create scatter plot and log to tensorboard (very slow)
        if do_scatter and (len(arrs) > 1):
            (k1, k2, *_), (y1, y2, *_) = zip(*arrs.items())

            def accumulate_batch(y1, y2, k1=k1, k2=k2, scope=scope) -> int:
                """ Gather the new batch outside of the tensorflow graph """
                if scope not in batches:
                    batches[scope] = {}
                scatters = batches[scope]
                scatters[k1] = scatters.get(k1, []) + [y1]
                scatters[k2] = scatters.get(k2, []) + [y2]
                return len(scatters[k1])

            # Accumulate outside of the TF context across graph executions
            tf.numpy_function(accumulate_batch, (y1, y2), tf.int64)

            if record:#if isinstance(record, bool) and record:
                # Can't contain '/'?
                # with keras.name_scope(f'scatter/{scope.replace("@",".")}'):
                with tf.name_scope(f'scatter/{scope.replace("@",".")}'):
                    image = lambda label, *y, **kw: tf.summary.image(label,
                        tf.numpy_function(partial(scatter, **kw), y, tf.uint8))
                    image('image', y1, y2, y1_label=k1, y2_label=k2, key=scope)

    return calculate_loss

