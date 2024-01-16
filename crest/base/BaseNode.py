from collections.abc import Callable, Collection
from functools import cached_property, partial
from contextlib import contextmanager 
from abc import abstractmethod
from scipy import stats

import matplotlib.pyplot as plt 
import tensorflow as tf 
import seaborn as sns
import numpy as np
import warnings, io

from crest.utils import classproperty
from crest.base  import BaseAbstract


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
    *normalize : tuple[Callable]
        A pair of functions defining (forward transform, inverse transform)
        for the input and output features, respectively. By default, no 
        pre- or post-processing is applied. 
    loss       : str | Callable | dict[str, str | Callable | dict]
        Loss function that should be used for this Model. This can be passed
        as a standard keras loss string (e.g. 'mse'); as an arbitrary callable
        which has the signature loss(y_true, y_pred) -> float; or, if multiple
        outputs are returned from this Model, a dictionary mapping output 
        feature names to the respective loss str or callable for that feature.
    transform_loss : bool
        Determines whether losses should be applied before or after the post-
        processing (if any exists). In other words, with transform_loss=True
        (the default), losses will be calculated on the transformed outputs
        of the model (prior to post-processing), rather than the inverse 
        transformed version. This can help avoid needing to weight model losses
        differently, when multiple sub-model losses are being used to train a
        larger hierarchal model. Note this has no effect if no *normalize
        functions are given. 
    debug          : bool
        Determines whether y_true/y_pred statistics should be printed on each
        batch inside the loss wrapper (default: False). Note this has no effect
        if no *normalize functions are given, or transform_loss=False. 

    """
    inputs  : IO_TYPE
    outputs : IO_TYPE 

    def __init__(self, 
        *normalize     : tuple[Callable], 
        loss           : str | Callable | dict[str, str | Callable | dict] = 'mse',
        transform_loss : bool = True,
        debug          : bool = False,
    ):
        self.normalize      = normalize
        self.loss           = loss
        self.debug          = debug
        self.transform_loss = transform_loss


    @abstractmethod
    def call(self, X: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:      
        """ Inheriting classes must define a `call` method, which takes
            as input a dictionary of {feature name: Input Tensor}, and
            returns a dictionary of {output feature: Output Tensor}. """
        raise NotImplementedError(f'{self}.call() must be implemented')


    @classproperty
    def input_spec(cls) -> dict[str, tf.TensorSpec]:
        """ Convert input shape dictionary into TensorSpec objects """
        return cls._generate_spec(cls.inputs)


    @classproperty
    def output_spec(cls) -> dict[str, tf.TensorSpec]:
        """ Convert output shape dictionary into TensorSpec objects """
        return cls._generate_spec(cls.outputs)


    @cached_property
    def graph(self) -> 'HierarchalTensorGraph':
        """ Create the default base node graph object """
        from crest.model import HierarchalTensorGraph as HTG
        return HTG(**{
            'node'    : self.call,
            'name'    : f'{self}', 
            'inputs'  : self.input_spec, 
            'outputs' : self.output_spec,
        })


    @property
    def losses(self) -> dict[str, Callable]:
        """ Return a dictionary of losses, one per output feature """
        get_loss = lambda loss: (loss if not isinstance(loss, (str, dict))
                                      else tf.keras.losses.get(loss))

        if isinstance(self.loss, dict) and ('class_name' not in self.loss):
            losses = {k: get_loss(loss) for k, loss in self.loss.items()}
        else:
            loss_f = get_loss(self.loss)
            losses = {feature: loss_f for feature in self.output_spec}

        # If pre/post processing added, need to allow calculating loss on
        # transformed model outputs rather than forcing the postprocessed
        if hasattr(self, 'postprocess') and self.transform_loss:
            def loss_wrapper(feature: str, loss_f: Callable):
                """ Calculate loss on the raw model outputs (and preprocessed
                    ground truth), rather than the postprocessed outputs. """

                def transform(y_true, y_pred):
                    def from_dict(z):
                        try:    return z[feature] if isinstance(z, dict) else z
                        except: raise Exception(f'Expected key {feature}: {z}')

                    # Helper that gathers various data statistics to print
                    stats = lambda y: [(k, getattr(tf, k)(y)) for k in [
                        'shape', 'reduce_min', 'reduce_max', 'reduce_mean']]

                    # Helper that logs a scatter plot via tensorboard
                    image = lambda label, *y, **kw: tf.summary.image(label,
                        tf.numpy_function(partial(scatter, **kw), y, tf.uint8))

                    if self.debug: tf.print('\nPrior to transform:',
                        '\ny_true', *stats(y_true),
                        '\ny_pred', *stats(y_pred))

                    # Create scatter plots both before and after postprocessing
                    with tf.name_scope(''): 
                        with tf.name_scope(f'{self}/{feature.replace("@",".")}'):
                            # Preprocess y_true, and gather raw model outputs
                            image('postprocessed', y_true, y_pred)
                            tf.summary.histogram('postprocessed/y_true', y_true)
                            tf.summary.histogram('postprocessed/y_pred', y_pred)
                            y_true = from_dict(self.preprocess({feature:y_true}))
                            y_pred = from_dict(self._call_output)
                            tf.summary.histogram('transformed/y_true', y_true)
                            tf.summary.histogram('transformed/y_pred', y_pred)
                            image('transformed', y_true, y_pred, color='g')

                    if self.debug: tf.print('\nAfter transform:',
                        '\ny_true', *stats(y_true),
                        '\ny_pred', *stats(y_pred),
                        '\nloss:',  *stats(loss_f(y_true, y_pred)))

                    return tf.reduce_mean( loss_f(y_true, y_pred) )
                return transform
            return {k: loss_wrapper(k, loss_f) for k, loss_f in losses.items()}
        return losses


    @staticmethod
    def _generate_spec(io_shapes_spec: IO_TYPE) -> dict[str, tf.TensorSpec]:
        """ Generate a TensorSpec dict from the given input/output config """

        def parse(coord_shapes: COORD_TYPE) -> list[int | None]:
            """ Convert window extent tuple to total shape """
            if not len(coord_shapes): return [None]
            # Handle left/right extent being used, and order by coordinate key
            get_total = lambda s: sum(s)+1 if isinstance(s, Collection) else s
            _,ordered = zip(*sorted(coord_shapes.items(), key=lambda kv:kv[0]))
            return [None] + list(map(get_total, ordered))

        return {feature: tf.TensorSpec(shape=parse(coord_shapes), name=feature)
                for source, feature_shapes in io_shapes_spec.items()
                for feature,  coord_shapes in feature_shapes.items()}


    def _add_data_transform(self, preprocess: Callable, postprocess: Callable):
        """ Add pre-/post-processing functions to the graph """
        assert(getattr(self, 'preprocess', None) is None), \
            f'Should only call {self}._add_data_transform once'

        self.preprocess  = preprocess
        self.postprocess = postprocess

        preprocess  = _NodeWrap(preprocess,  f'{self}-preprocess')
        call_output = _NodeWrap(self,        f'{self}-call')
        postprocess = _NodeWrap(postprocess, f'{self}-postprocess')

        i_spec = self.input_spec
        o_spec = self.output_spec

        # Instantiate graphs for each node to define input/output specs
        from crest.model import HierarchalTensorGraph as HTG
        preprocess  = HTG(preprocess,  'preprocess',  i_spec, i_spec)
        model       = HTG(call_output, f'{self}.call',preprocess.outputs, o_spec)
        postprocess = HTG(postprocess, 'postprocess', model.outputs, model.outputs)

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
        X = getattr(X, 'copy', lambda: X)()
        Y = {source: {k: X.pop(k) for k, coords in features.items()} 
                for source, features in self.inputs.items()}
        
        # Wrap output(s) with output_spec dictionary if not already
        out = self.call(Y | X)
        if not isinstance(out, (dict, list, tuple)):
            out = [out]
        if isinstance(out, (list, tuple)):
            if len(out) != len(self.output_spec):
                raise Exception(f'Expected {len(self.output_spec)} outputs'
                        + f' from {self}, but received {len(out)} value(s)')
            out = dict(zip(self.output_spec.keys(), out))

        def match_shape(feature):
            """ Match output shapes to the respective output_spec shape """
            squeezed  = tf.squeeze(out[feature])
            tgt_shape = self.output_spec[feature].shape
            rank_diff = tgt_shape.rank-tf.rank(squeezed)
            extra_dim = tf.ones(tf.math.abs(rank_diff), dtype=tf.int32)
            new_shape = tf.concat([tf.shape(squeezed), extra_dim], 0)
            matched   = tf.reshape(squeezed, shape=new_shape)
            return tf.ensure_shape(matched, tgt_shape)

        # Ensure output shapes match their output spec shape
        self._call_output = dict(zip(out, map(match_shape, out)))
        return self._call_output


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
            self._add_data_transform(*self.normalize)



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
        keywords = ['tensorflow', 'keras']
        for k, v in obj.__dict__.items():
            if any(name in str(type(v)) for name in keywords):
                setattr(self, k, v)
        self.obj = obj


    def __repr__(self): 
        return repr(self.obj)


    def call(self, X, *args, **kwargs):
        """ Adds output histograms for tensorboard visualization """
        output = getattr(self.obj, '_call', self.obj)(X, *args, **kwargs)

        with tf.name_scope(''):
            with tf.name_scope(self.name):

                # Add histograms for all outputs
                if isinstance(output, dict):
                    for k,v in output.items():
                        tf.summary.histogram(k.replace('@', '.'), output[k])
                else: tf.summary.histogram('output', output)
        
            # Add histograms for all weights
            for w in self.weights: 
                tf.summary.histogram(w.name, w)
        return output


    def get_config(self):
        return super().get_config() | {'obj': self.obj}


def scatter(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
    """ Create a scatter plot using y true and pred, returning as an array """
    ys = [y_true.flatten(), y_pred.flatten()]
    r2 = stats.linregress(*ys)[2] ** 2

    with plot_to_array() as array:
        g = sns.jointplot(x=ys[1], y=ys[0])
        g.plot_joint(sns.kdeplot, color='r', zorder=1, levels=6, alpha=0.8)
        # g.plot_marginals(sns.rugplot, color='r', height=-0.15, clip_on=False)
        plt.xlabel('y_pred')
        plt.ylabel('y_true')
        plt.title(rf'$r^{{2}}$ = {r2:.2f}   N = {len(ys[0])}', fontsize=15)
        plt.axline((0,0), slope=1, ls='--', color='k', alpha=0.5, zorder=2)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(-2,3))

        minim = min(plt.xlim()[0], plt.ylim()[0])
        maxim = max(plt.xlim()[1], plt.ylim()[1])
        plt.xlim((minim, maxim))
        plt.ylim((minim, maxim))
        return array


@contextmanager
def plot_to_array(dpi: int = 150, width: float = 4.8, height: float = 3.6):
    """ Save a matplotlib plot to a numpy array. 
    
    Examples
    --------
    >>> with plot_to_array() as array:
    ...     plt.scatter([1, 2], [2, 1])
    >>> plt.imshow(array)

    """
    width = 4.8
    height = 3.6
    shape = (1, int(height*dpi), int(width*dpi), 4)
    array = np.empty(shape, dtype=np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield array
    
        # Save the current figure into the previously returned array object
        with io.BytesIO() as buffer:
            figure = plt.gcf()
            try:

                plt.tight_layout()
                figure.set_size_inches([width, height])
                figure.canvas.draw() 
                figure.savefig(buffer, format='raw', dpi=dpi)
                figure.clf()
                plt.close(figure)
                plt.close('all')
                buffer.seek(0)
                array.ravel()[:] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            except:
                print('\n\n')
                print(figure.get_size_inches())
                print(figure.bbox)
                # raise