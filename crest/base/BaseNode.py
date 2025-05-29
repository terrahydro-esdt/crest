from collections.abc import Callable, Collection
from sklearn.metrics import r2_score
from scipy.stats import linregress
from functools import cached_property, partial, reduce
from operator import and_
from abc import abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import tlz
import json
import os
import importlib

from ..utils import classproperty, plot_to_array
from . import BaseAbstract


# BaseNode.inputs/outputs type annotation. These two dictionaries should
# follow the format of:
# { '<SOURCE>' :         # String indicating where features are found
#   { '<FEATURE>' :      # String indicating features that are needed
#     { '<COORDINATE>' : # String indicating coordinate dimensions, with either:
#       total            # - int, giving the total size of windows along this coordinate
#     | None             # - None, indicating windows can be any size (can be set as a model hyperparameter)
#     | (left, right)    # - Collection of two integers, defining left/right window extent (total=left+right+1)
# } } }
#
# _coord_type defines the inner coordinate dictionary annotation
COORD_TYPE = dict[str, Collection[int] | int | None]

# IO_TYPE defines the full inputs/outputs annotation
IO_TYPE = dict[str, dict[str, COORD_TYPE]]


def _serialize_callable(obj):
    if callable(obj):
        return obj.__name__ if hasattr(obj, '__name__') else str(obj)
    return obj


class BaseNode(BaseAbstract):
    """BaseNode provides a template for machine learning models to inherit from. 

    Notes
    -----
    Any Model(BaseNode) definition must define Model.inputs and Model.outputs, 
    i.e. class-level dictionary attributes. These dictionaries define the input
    and output shapes for features coming into, and going out of, the model. 
    BaseNode objects will then automatically have input_spec and output_spec 
    attributes defined, allowing the specs to be directly passed into any 
    HierarchalTensorGraph definition. As well, if forward/inverse normalization
    functions are given to the Model upon initialization, pre- and post-
    processing nodes will be added to the Model, which will normalize and 
    de-normalize the inputs and outputs of the Model, respectively.

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
        larger hierarchal model. Note this has no effect if no *normalize
        functions are given. 
        Alternatively, a tuple of callables (in the same format as normalize)
        can be given to use as a different normalization procedure than the 
        feature normalization.
    debug          : bool
        Determines whether y_true/y_pred statistics should be printed on each
        batch inside the loss wrapper (default: False). Note this has no effect
        if no *normalize functions are given, or transform_loss=False. 
    plot_scatter   : bool
        Determines whether scatter plots should be created in the loss function
        to add to TensorBoard. Setting to True will add target vs predicted 
        scatter plots for each loss function update, which in turn slows down
        the training process.

    """
    inputs: IO_TYPE
    outputs: IO_TYPE

    def __init__(self, 
        normalize      = (), #: Union[tuple[Callable, Callable], 'Transform'] = (), 
        loss           : str | Callable | dict[str, str | Callable | dict] = 'mse',
        transform_loss : bool | tuple[Callable, Callable] = True,
        debug          : bool = False,
        plot_scatter   : bool = False,
    ):
        self.normalize      = getattr(normalize, 'by_feature', normalize)
        self.loss           = loss
        self.debug          = debug
        self.transform_loss = transform_loss
        self.plot_scatter = plot_scatter

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def call(self, X: dict[str, tf.Tensor], training: bool) -> dict[str, tf.Tensor]:      
        """ Inheriting classes must define a `call` method, which takes
            as input a dictionary of {feature name: Input Tensor}, and
            returns a dictionary of {output feature: Output Tensor}. """
        raise NotImplementedError(f'{self}.call() must be implemented')

    @cached_property
    def graph(self) -> 'HierarchalTensorGraph':
        """ Create the default base node graph object """
        from crest.model import HierarchalTensorGraph as HTG
        return HTG(**{
            'node': _NodeWrap(self, f'{self}-call'),
            'name': f'{self}',
            'inputs': self.input_spec,
            'outputs': self.output_spec,
        })

    @property
    def losses(self) -> dict[str, Callable]:
        """ Return a dictionary of losses, one per output feature """
        def get_loss(loss): return (loss if not isinstance(loss, (str, dict))
                                    else tf.keras.losses.get(loss))

        # Allow loss specification via config dict (see tf.keras.losses.get)
        if isinstance(self.loss, dict) and ('class_name' not in self.loss):
            losses = {k: get_loss(loss) for k, loss in self.loss.items()}
        else:
            losses = {k: get_loss(self.loss) for k in self.output_spec}

        wrap_loss = partial(loss_wrapper, **{
            'scope': f'{self}',
            'transform': getattr(self, 'loss_transformer', None),
            'postprocess': getattr(self, 'postprocess', None),
            'plot_scatter': self.plot_scatter,
            'debug': self.debug,
        })

        # Create a wrapper for loss functions to enable masking NaN
        # targets/predictions, and applying different normalization
        return dict(zip(losses, map(wrap_loss, losses, losses.values())))

    @classproperty
    def input_spec(cls) -> dict[str, tf.TensorSpec]:
        """ Convert input shape dictionary into TensorSpec objects """
        return cls._generate_spec(cls.inputs)

    @classproperty
    def output_spec(cls) -> dict[str, tf.TensorSpec]:
        """ Convert output shape dictionary into TensorSpec objects """
        return cls._generate_spec(cls.outputs)

    @staticmethod
    def _generate_spec(io_shapes_spec: IO_TYPE) -> dict[str, tf.TensorSpec]:
        """ Generate a TensorSpec dict from the given input/output config """

        def parse(coord_shapes: COORD_TYPE) -> list[int | None]:
            """ Convert window extent tuple to total shape """
            if not len(coord_shapes):
                return [None]
            # Handle left/right extent being used, and order by coordinate key
            def get_total(s): return sum(
                s)+1 if isinstance(s, Collection) else s
            _, ordered = zip(
                *sorted(coord_shapes.items(), key=lambda kv: kv[0]))
            return [None] + list(map(get_total, ordered))

        return {feature: tf.TensorSpec(shape=parse(coord_shapes), name=feature)
                for source, feature_shapes in io_shapes_spec.items()
                for feature,  coord_shapes in feature_shapes.items()}

    def _add_data_transform(self):
        """ Add pre-/post-processing functions to the graph """
        assert (getattr(self, 'preprocess', None) is None), \
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

        preprocess = _NodeWrap(self.preprocess,  f'{self}-preprocess')
        call_output = _NodeWrap(self,             f'{self}-call')
        postprocess = _NodeWrap(self.postprocess, f'{self}-postprocess')

        i_spec = self.input_spec
        o_spec = self.output_spec

        # Instantiate graphs for each node to define input/output specs
        from crest.model import HierarchalTensorGraph as HTG
        preprocess = HTG(preprocess,  'preprocess',   i_spec, i_spec)
        model = HTG(call_output, f'{self}.call', preprocess.outputs, o_spec)
        postprocess = HTG(postprocess, 'postprocess',
                          model.outputs, model.outputs)

        # Reset graph so that it isn't a base node
        self.graph.node = self.graph
        self.graph.inputs = preprocess.inputs
        self.graph.outputs = postprocess.outputs

        # Add new edges for pre-/post-processing
        self.graph.add_edges_from([
            ('input', preprocess),
            (preprocess, model),
            (model, postprocess),
            (postprocess, 'output'),
        ])

    def _call(self, X, training=False):
        """ Wraps self.call to provide data parsing / shape validation """
        # Group features by source
        Z = getattr(X, 'copy', lambda: X)()
        Y = {source: {k: Z.pop(k) for k, coords in features.items()}
             for source, features in self.inputs.items()}
        X = Z | Y

        for source, features in X.items():
            if isinstance(features, dict):
                # Handle any coordinate features
                X[source] = self.convert_coords(features)

                # One-hot encode various class features
                X[source] = self.convert_onehot(features, {
                    'landcover_class_0': (17, None),
                    'slt': (7, None),
                    'tvl': (20, [0, 1, 6, 8, 9, 10, 12, 15, 16, 19]),
                    'tvh': (20, [2, 3, 4, 5, 17, 18]),
                })

        # Wrap output(s) with output_spec dictionary if not already
        out = self.call(X, training)
        if not isinstance(out, (dict, list, tuple)):
            out = [out]
        if isinstance(out, (list, tuple)):
            if len(out) != len(self.output_spec):
                raise Exception(f'Expected {len(self.output_spec)} outputs'
                                + f' from {self}, but received {len(out)} value(s)')
            out = dict(zip(self.output_spec.keys(), out))

        def match_shape(feature):
            """ Match output shapes to the respective output_spec shape """
            # TODO: Handle case where tgt_shape has internal dimensions sized 1
            #       e.g. tgt_shape=(64,1,4,4) will cause this to fail even if
            #            the given out tensor has the same shape initially, as
            #            it will be squeezed and reshaped to (64,4,4,1)
            squeezed = tf.squeeze(out[feature])
            tgt_shape = self.output_spec[feature].shape
            rank_diff = tgt_shape.rank-tf.rank(squeezed)
            extra_dim = tf.ones(tf.math.abs(rank_diff), dtype=tf.int32)
            new_shape = tf.concat([tf.shape(squeezed), extra_dim], 0)
            matched = tf.reshape(squeezed, shape=new_shape)
            return tf.ensure_shape(matched, tgt_shape)

        # Ensure output shapes match their output spec shape
        return dict(zip(out, map(match_shape, out)))

    def __call__(self, X):
        """ Calling a BaseNode passes input through its graph """
        return self.graph(getattr(X, 'copy', lambda: X)())

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
            assert (count and funcs), f'Arguments to {self} should be two ' + \
                'callables which define forward and inverse transformations'
            self._add_data_transform()

        # Set the loss transformer if only it was given
        elif not isinstance(self.transform_loss, bool):

            # Allow just a single transformer to be given (rather than
            # requiring both forward/inverse, since inverse is not used)
            if isinstance(self.transform_loss, (list, tuple)):
                self.loss_transformer = self.transform_loss[0]
            else:
                self.loss_transformer = self.transform_loss
            assert (callable(self.loss_transformer))

        # self.normalize needs to be set if transform_loss = True
        elif self.transform_loss:
            raise Exception('Cannot transform loss if no functions are given')

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
                solve += f'initialize Transformer; or, set transform_loss=False'

                # If no outputs can be transformed, raise an exception
                if len(unable) == len(self.output_spec):
                    raise Exception(f'{error} any of the outputs!\n{solve}')

                # Otherwise just print a warning
                print(f'WARNING: {error} some outputs: {unable}\n{solve}')

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
                    values = tf.cast(values, tf.int32)
                onehot = tf.one_hot(values, total_cls)
                for i in include_cls or range(total_cls):
                    X[f'{feature}_{i}'] = onehot[..., i]
        return X

    def convert_coords(self, X: dict) -> dict:
        """ Reshape coordinates to be stackable alongside other features """
        coord_names = ['datetime', 'latitude', 'longitude']
        coord_shape = tf.shape(tlz.first(tlz.dissoc(X, *coord_names).values()))

        # Expected shape of other features is [batch, datetime, lat, lon]. If
        # we don't find a tensor with four dimensions, just return the original
        # dict. Better handling for other tensor shapes (e.g. static) should be
        # implemented in the future.
        if coord_shape.shape[0] != 4: return X

        ndt, nlt, nln = coord_shape[1], coord_shape[2], coord_shape[3]
        for name in coord_names:
            if name in X:

                # Tile coordinate features into 3d cube
                coordinate = X.pop(name)
                tiles, idx = {
                    'datetime': ([1, 1, nlt, nln], [slice(None), slice(None), None, None]),
                    'latitude': ([1, ndt, 1, nln], [slice(None), None, slice(None), None]),
                    'longitude': ([1, ndt, nlt, 1], [slice(None), None, None, slice(None)]),
                }[name]

                if name == 'datetime':
                    # Convert datetime to value between [0, seconds in 24 hours]
                    for mult in [1, 365]:
                        sec_in_day = 60 * 60 * 24 * mult
                        coord_secs = 60 * tf.cast(coordinate, tf.float64) # datetime[m] -> seconds
                        coord_secs = tf.math.floormod(coord_secs, sec_in_day)
    
                        with tf.name_scope(''):
                            tf.summary.histogram(f'{self}-call/datetime_{mult}_laststep', coord_secs[:, -1])
                            tf.summary.histogram(f'{self}-call/datetime_{mult}_lastsample', coord_secs[-1])
    
                        # Convert to radians
                        coord_rads = tf.cast(coord_secs * 2 * np.pi / sec_in_day, tf.float32)
            
                        # Convert datetime to a set of two periodic sin/cos features
                        for name in ['sin', 'cos']:
                            sin_cos_dt = getattr(tf.math, name)(coord_rads)
    
                            with tf.name_scope(''):
                                tf.summary.histogram(f'{self}-call/datetime_{mult}_laststep_{name}', sin_cos_dt[:, -1])
                                tf.summary.histogram(f'{self}-call/datetime_{mult}_lastsample_{name}', sin_cos_dt[-1])
                            X[f'{name}_{mult}'] = tf.tile(sin_cos_dt[idx], tiles)
                else: X[name] = tf.tile(coordinate[idx], tiles)
        return X

    # def to_json(self) -> str:
    #     """ Serialize the BaseNode instance to JSON """
    #     config = {
    #         'class_name': self.__class__.__name__,
    #         'module': self.__class__.__module__,
    #         'inputs': self.inputs,
    #         'outputs': self.outputs,
    #         'loss': _serialize_callable(self.loss),
    #         'transform_loss': tuple(map(_serialize_callable, self.transform_loss)) if isinstance(self.transform_loss, (tuple, list)) else self.transform_loss,
    #         'normalize': tuple(map(_serialize_callable, self.normalize)) if self.normalize else None,
    #         'debug': self.debug,
    #         'plot_scatter': self.plot_scatter,
    #     }
    #     return json.dumps(config)

    # def from_json(cls, json_str: str) -> 'BaseNode':
    #     """ Deserialize a BaseNode instance from a JSON string """
    #     config = json.loads(json_str)

    #     # Dynamically import class
    #     module = importlib.import_module(config['module'])
    #     class_ = getattr(module, config['class_name'])

    #     # Reconstruct any standard loss functions or callables
    #     def resolve(obj):
    #         if isinstance(obj, str):
    #             try:
    #                 return getattr(tf.keras.losses, obj)
    #             except AttributeError:
    #                 try:
    #                     return globals()[obj]
    #                 except KeyError:
    #                     return obj  # fallback
    #         return obj

    #     loss = resolve(config['loss'])
    #     normalize = tuple(
    #         map(resolve, config['normalize'])) if config['normalize'] else ()
    #     transform_loss = config['transform_loss']
    #     if isinstance(transform_loss, (list, tuple)):
    #         transform_loss = tuple(map(resolve, transform_loss))
    #     else:
    #         transform_loss = resolve(transform_loss)

    #     return class_(
    #         *normalize,
    #         loss=loss,
    #         transform_loss=transform_loss,
    #         debug=config['debug'],
    #         plot_scatter=config['plot_scatter']
    #     )

    # def save(self, path: str) -> None:
    #     """Save the node configuration to a JSON file."""
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     with open(path, 'w') as f:
    #         f.write(self.to_json())

    # @classmethod
    # def load(cls, path: str) -> 'BaseNode':
    #     """Load a node configuration from a JSON file."""
    #     with open(path, 'r') as f:
    #         return cls.from_json(f.read())


class _NodeWrap(tf.keras.layers.Layer):
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

    def __init__(self, obj: Callable, name: str = ''):
        super().__init__(name=name or getattr(obj, 'name', str(obj)))

        # Add all tensorflow/keras objects to allow weight tracking
        keywords = ['tensorflow', 'keras', 'terrahydro']
        for k, v in obj.__dict__.items():
            if any(name in str(type(v)) for name in keywords):
                setattr(self, k, v)

        self.obj = obj
        self.obj_name = getattr(obj, '__name__', repr(obj))

    def __repr__(self):
        return repr(self.obj)

    @property
    def __name__(self):
        return f'_NodeWrap({self.obj.__class__.__name__})'

    def call(self, X, *args, **kwargs):
        """ Adds output histograms for tensorboard visualization """
        with tf.name_scope(''):
            with tf.name_scope(self.name):
                self._add_histograms(X, 'input')
                out = getattr(self.obj, '_call', self.obj)(X, *args, **kwargs)
                self._add_histograms(out, 'output', desc=self.obj_name)

            # Add histograms for all weights
            for w in self.weights:
                tf.summary.histogram(w.name, w)
        return out

    def get_config(self):
        return super().get_config() | {
            'obj': self.obj,
            'obj_name': self.obj_name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _add_histograms(self, X, scope: str, desc: str | None = None):
        """ Log the given X dict/tensor(s) histograms under the given scope """
        if isinstance(X, dict):
            for k, v in X.items():
                label = k.replace('@', '.') + f'/{scope}'
                tf.summary.histogram(label, tf.identity(v), description=desc)
        else:
            tf.summary.histogram(scope, tf.identity(X), description=desc)


def loss_wrapper(
    feature: str,
    loss_func: Callable,
    scope: str = '',
    transform: Callable | None = None,
    postprocess: Callable | None = None,
    plot_scatter: bool = False,
    debug: bool = False,
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

        def mask_nans(*arrs) -> tuple:
            """ Mask the union of NaN indices for given arrays """
            # Extract feature value if a dict is given, then mask NaNs
            arrs = [a[feature] if isinstance(a, dict) else a for a in arrs]
            mask = reduce(and_, map(tf.math.is_finite, arrs))
            return tuple((tf.boolean_mask(a, mask) for a in arrs))

        # Sanity check that y_true and y_pred have the same sizes
        tf.debugging.assert_equal(tf.size(y_true), tf.size(y_pred))
        
        # Select the requested feature and valid samples
        arrs = [a[feature] if isinstance(a, dict) else a for a in [y_true, y_pred]]               
        mask = reduce(and_, map(tf.math.is_finite, arrs))
        flag = tf.math.reduce_any(mask)

        # Label the targets/predictions with their transformation
        y_label = lambda y,f: y if f is None else f'{getattr(f,"__name__",f)}({y})'

        # Mask NaNs in both targets and predictions, and log results
        masked = mask_nans(y_true, y_pred)
        labels = ['y_true', y_label('y_pred', postprocess)]
        logging(f'{scope}/{feature}/original/', dict(zip(labels, masked)))

        # Preprocess the target if we want to calculate normalized loss
        if transform is not None:
            y_true = transform({feature: y_true})
            y_pred = transform({feature: y_pred})

            arrs = [a[feature] if isinstance(a, dict) else a for a in [
                y_true, y_pred]]
            mask = reduce(and_, map(tf.math.is_finite, arrs))
            flag = tf.math.reduce_any(mask)

            masked = mask_nans(y_true, y_pred)
            labels = [y_label('y_true', transform), 'y_pred']
            logging(f'{scope}/{feature}/transform/', dict(zip(labels, masked)))

        # Calculate and log the final loss, then return if any exist
        loss = loss_func(*masked)
        zero = tf.size(loss) == 0
        logging(f'{scope}/{feature}/loss/', {'loss': loss}, False)
        # return tf.cond(zero, lambda: 0., lambda: tf.reduce_mean(loss))
        return tf.cond(flag, lambda: tf.reduce_mean(loss), lambda: 0.)

    # Defines various logging functionality in a separate helper method
    def logging(scope: str, arrs: dict, do_scatter=plot_scatter) -> None:
        """ Output debug logs, scatter plots, histograms """

        def scatter(y1: np.ndarray, y2: np.ndarray, **kwargs) -> np.ndarray:
            """ Create scatter plot and return the image as a numpy array """
            ys = y1, y2 = y1.flatten(), y2.flatten()
            try:
                r2 = linregress(*ys)[2] ** 2  # pyright: ignore
            except:
                r2 = np.nan
            try:
                R2 = r2_score(*ys)
            except:
                R2 = np.nan

            # Write the plot image to the array variable
            with plot_to_array() as array:
                # Scatter plot with KDE contours
                sns.jointplot(x=y2, y=y1).plot_joint(sns.kdeplot,
                                                     color='r', zorder=1, levels=6, alpha=0.7)

                # 1:1 diagonal line
                plt.axline((0, 0), slope=1, ls='--',
                           color='k', alpha=0.5, zorder=2)

                # Plot labels
                plt.ticklabel_format(
                    style='sci', axis='both', scilimits=(-2, 3))
                plt.ylabel(kwargs.get('y1_label', 'y_true'), fontsize=14)
                plt.xlabel(kwargs.get('y2_label', 'y_pred'), fontsize=14)
                plt.title('  '.join([
                    rf'$R^{{2}}$ = {R2:.2f}',
                    rf'$r^{{2}}$ = {r2:.2f}',
                    f'N = {len(y1)}',
                ]), fontsize=14)

                # Use the same extent limits for both axes
                minim = min(plt.xlim()[0], plt.ylim()[0])
                maxim = max(plt.xlim()[1], plt.ylim()[1])
                plt.xlim((minim, maxim))
                plt.ylim((minim, maxim))
                return array

        # Print various statistics for debugging
        if debug:
            align = max(map(len, arrs))

            def stats(y): return [(k, getattr(tf, k)(y)) for k in [
                'shape', 'reduce_min', 'reduce_max', 'reduce_mean']]
            tf.print(f'\n\n{scope}:', *[output
                                        for k, arr in arrs.items()
                                        for output in [f'\n{k:>{align}}'] + stats(arr)])

        with tf.name_scope(scope.replace("@", ".")):
            # Log array histograms to tensorboard
            for k, v in arrs.items():
                tf.summary.histogram(k, v)

        with tf.name_scope(f'scatter/{scope.replace("@",".")}'):
            # Create scatter plot and log to tensorboard (very slow)
            if do_scatter: 
                (k1, k2, *_), (y1, y2, *_) = zip(*arrs.items())
                image = lambda label, *y, **kw: tf.summary.image(label,
                                                                 tf.numpy_function(partial(scatter, **kw), y, tf.uint8))
                image('image', y1, y2, y1_label=k1, y2_label=k2)

    return calculate_loss
