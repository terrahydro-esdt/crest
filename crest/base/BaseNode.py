from collections.abc import Callable, Collection
from functools import cached_property, partial
from abc import abstractmethod
from scipy import stats

import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np
import io 

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
    def call(self, X):      
        raise NotImplementedError(f'{self}.call() must be implemented')


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
        get_loss = lambda loss: tf.keras.losses.get(loss) \
                            if isinstance(loss, (str, dict)) else loss 

        if isinstance(self.loss, dict) and ('class_name' not in self.loss):
            losses = {k: get_loss(loss) for k, loss in self.loss.items()}
        else:
            loss_f = get_loss(self.loss)
            losses = {feature: loss_f for feature in self.output_spec}

        # If pre/post processing added, need to allow calculating loss on
        # transformed model outputs rather than forcing the postprocessed
        if hasattr(self, 'postprocess') and self.transform_loss:
            def loss_wrapper(key: str, loss_f: Callable):
                """ Calculate loss on the raw model outputs (and preprocessed
                    ground truth), rather than the postprocessed outputs. """
                def transform(y_true, y_pred):
                    def from_dict(z):
                        try:    return z[key] if isinstance(z, dict) else z
                        except: raise Exception(f'Expected key {key}: {z}')

                    keys  = ['shape', 'reduce_min', 'reduce_max']
                    stats = lambda y:[f'{k}: {getattr(tf,k)(y)}' for k in keys]

                    if self.debug:
                        tf.print('\nPrior to transform:',
                            '\ny_true', '\t'.join(stats(y_true)),
                            '\ny_pred', '\t'.join(stats(y_pred)) )

                    with tf.name_scope(''):
                        gen_img = lambda *y, **kw: tf.numpy_function(partial(scatterplot, **kw), y, tf.uint8, stateful=False)
                        tf.summary.image(f'{self}/{key}/postprocessed', gen_img(y_true, y_pred))

                    # Preprocess y_true features, and gather
                    # model outputs prior to postprocessing
                    y_true = from_dict( self.preprocess({key: y_true}) )
                    y_pred = from_dict( self._call_output )

                    with tf.name_scope(''):
                        tf.summary.image(f'{self}/{key}/transformed', gen_img(y_true, y_pred, color='g'))

                    if self.debug:
                        tf.print('\nAfter transform:',
                            '\ny_true', '\t'.join(stats(y_true)),
                            '\ny_pred', '\t'.join(stats(y_pred)) )
                        
                    return tf.reduce_mean( loss_f(y_true, y_pred) )
                return transform
            return {k: loss_wrapper(k, loss_f) for k, loss_f in losses.items()}
        return losses


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
        preprocess  = HTG(preprocess,  'preprocess',   i_spec, i_spec)
        model       = HTG(call_output, f'{self}.call', i_spec, o_spec)
        postprocess = HTG(postprocess, 'postprocess',  o_spec, o_spec)

        # Reset graph so that it isn't a base node
        self.graph.node = self.graph

        # Add new edges for pre-/post-processing
        self.graph.add_edges_from([
            ('input', preprocess),
            (preprocess, model),
            (model, postprocess),
            (postprocess, 'output'),
        ])


    def _call(self, X):
        """ Wraps self.call to store the output tensor in self._call_output """
        self._call_output = self.call(getattr(X, 'copy', lambda: X)())
        return self._call_output


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
        output = getattr(self.obj, '_call', self.obj)(X)

        # Add histograms for all outputs
        with tf.name_scope(''):
            if isinstance(output, dict):
                for k,v in output.items():
                    tf.summary.histogram(f'{self.name}/output/{k}', output[k])
            else: tf.summary.histogram(f'{self.name}/output', output)
        return output

    def get_config(self):
        return super().get_config() | {'obj': self.obj}


def scatterplot(y_true, y_pred, **kwargs):
    def add_identity(ax, *line_args, **line_kwargs):
        ''' 
        Add 1 to 1 diagonal line to a plot.
        https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
        
        Usage: add_identity(plt.gca(), color='k', ls='--')
        '''
        line_kwargs['label'] = line_kwargs.get('label', '_nolegend_')
        identity, = ax.plot([], [], *line_args, **line_kwargs)
        
        def callback(axes):
            low_x, high_x = ax.get_xlim()
            low_y, high_y = ax.get_ylim()
            lo = max(low_x,  low_y)
            hi = min(high_x, high_y)
            identity.set_data([lo, hi], [lo, hi])

        callback(ax)
        ax.callbacks.connect('xlim_changed', callback)
        ax.callbacks.connect('ylim_changed', callback)

    with io.BytesIO() as buf:
        f = plt.Figure()
        ax = f.gca()
        ax.scatter(y_pred, y_true, **kwargs)
        ax.set_xlabel('y_pred')
        ax.set_ylabel('y_true')
        add_identity(ax, ls='--', color='k', alpha=0.5)
        slope_, intercept_, r_value, p_value, std_err = stats.linregress(y_true.flatten(), y_pred.flatten())
        ax.set_title(f'R^2 = {r_value**2:.2f}', fontsize=14)

        # Return plot as RGBA array
        f.canvas.draw() 
        f.savefig(buf, format='raw')
        buf.seek(0)
        return np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8), (1, int(f.bbox.bounds[3]), int(f.bbox.bounds[2]), -1))