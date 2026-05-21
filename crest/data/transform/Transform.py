from __future__ import annotations
from functools import cached_property, wraps, partial
from typing import Union, Callable
from scipy.stats import norm

import tensorflow as tf
import xarray as xr
import numpy as np
import traceback

from crest.base import BaseAbstract
from crest.utils import classproperty


class Transform(BaseAbstract):
    """ Class to enable easy data transformations.

    Parameters
    ----------
    statistics : xr.DataArray
        Summary statistics for all features that should be transformed.
    by_feature : dict
        Dictionary defining transformations to apply by feature name; i.e.
        `{feature_name: transformation}`. Keys are strings referencing a given
        feature by name, with '*' allowed to be used as a special key denoting
        a catch-all transformation to be applied when a given feature is not
        otherwise contained in the dictionary. Values may be defined as:

        - a string, referencing a predefined transformation contained in
          this class (e.g. 'normalize');
        - a function, directly giving the callable that should be applied,
          which should have the signature `function(data, key)`;
        - a Sequence of two elements, where each element can be one of the
          two previously stated types (string or function) and representing
          the same. The first element will be applied during the forward
          transformation of a feature, and the second element during the
          inverse transformation of that feature. Generally, the following
          relation should hold: `inverse(forward(data)) == data`.
 
    drop : list[str]
        Features which have entries within the summary statistics array,
        but should be excluded from the transformation process (e.g. for
        features representing classes / categories / flags).
    eps  : Number
        Small number used to avoid calculating NaN or inf.  

    """
    def __init__(self, 
        statistics : xr.DataArray, 
        by_feature : dict = {},
        drop : list[str]  = [], 
        eps  : Union[int, float] = 1e-6,
    ):
        self.stats = statistics.where(~statistics.features.isin(drop), drop=True)
        self.drop  = drop
        self.eps   = eps
        self.feature_transforms = by_feature


    def copy(self, **kwargs) -> 'Transform':
        """ Return a copy of this Transform, updating any given attributes """
        return Transform(**({
            'statistics' : self.stats.copy(),
            'by_feature' : dict(self.feature_transforms),
            'drop' : list(self.drop),
            'eps'  : self.eps,
        } | kwargs))


    @classproperty
    def available_transforms(self) -> list[str]:
        """ Return a list containing the names of all available transforms """
        is_transform = lambda s: s[:2]!='__' and hasattr(self, f'forward_{s}')
        return [s for s in dir(self) if is_transform(s)]


    # Available transformations:
    #   - by_feature (requires Transform to be initialized with by_feature)
    #   - logexp
    #   - logexpp1
    #   - standardize
    #   - robust
    #   - normalize
    #   - iqrnorm
    #   - adaptnorm
    #   - quantile
    # 
    # TODO: Refactor so that these transformations are in separate files
    #       contained within a transformations/ folder
    # ==========================
    def forward_by_feature(self, data, **kwargs): 
        """ Apply forward transform to a feature """
        return self._process_feature(data, index=0, **kwargs)

    def inverse_by_feature(self, data, **kwargs):
        """ Apply inverse transform to a feature """
        return self._process_feature(data, index=1, **kwargs)

    @property
    def by_feature(self):
        """ Return forward and inverse transforms for features """
        assert(self.feature_transforms), 'Must define feature_transforms'
        return ( self._map_transform(self.forward_by_feature), 
                 self._map_transform(self.inverse_by_feature) )


    # Log transform
    def forward_logexp(self, data, **kwargs):
        base = tf.math if isinstance(data, tf.Tensor) else np
        return base.log(data + self.eps)

    def inverse_logexp(self, data, **kwargs): 
        base = tf.math if isinstance(data, tf.Tensor) else np
        return base.exp(data) - self.eps

    @property
    def logexp(self): 
        return ( self._map_transform(self.forward_logexp), 
                 self._map_transform(self.inverse_logexp) )

    
    # Log(data+1) transform
    def forward_logexpp1(self, data, **kwargs):
        base = tf.math if isinstance(data, tf.Tensor) else np
        return base.sign(data) * base.log1p(base.abs(data)+self.eps)

    def inverse_logexpp1(self, data, **kwargs): 
        base = tf.math if isinstance(data, tf.Tensor) else np
        return base.sign(data) * (base.expm1(base.abs(data))-self.eps)

    @property
    def logexpp1(self): 
        return ( self._map_transform(self.forward_logexpp1), 
                 self._map_transform(self.inverse_logexpp1) )


    # 0 mean, 1 standard deviation data standardization
    def forward_standardize(self, data, **kwargs): 
        return (data - self.mean) / (self.std + self.eps)

    def inverse_standardize(self, data, **kwargs): 
        return data * (self.std + self.eps) + self.mean

    @property
    def standardize(self): 
        return ( self._map_transform(self.forward_standardize), 
                 self._map_transform(self.inverse_standardize) )


    # Interquartile range scaling
    def forward_robust(self, data, **kwargs): 
        return (data - self.median) / ((self.p75 - self.p25) + self.eps)

    def inverse_robust(self, data, **kwargs): 
        return data * ((self.p75 - self.p25) + self.eps) + self.median

    @property
    def robust(self): 
        return ( self._map_transform(self.forward_robust), 
                 self._map_transform(self.inverse_robust) )


    # Map data values to a given range; default [-1, 1]
    def forward_normalize(self, data, minim=-1, maxim=1, **kwargs):
        scaled = (data - self.min) / ((self.max - self.min) + self.eps)
        return scaled * (maxim - minim) + minim
    
    def inverse_normalize(self, data, minim=-1, maxim=1, **kwargs):
        scaled = ((data - minim) / (maxim - minim))
        return scaled * ((self.max - self.min) + self.eps) + self.min
    
    @property
    def normalize(self): 
        return ( self._map_transform(self.forward_normalize), 
                 self._map_transform(self.inverse_normalize) )


    # Map data quartiles to a given range; default [-1, 1]
    def forward_iqrnorm(self, data, minim=-1, maxim=1, **kwargs):
        scaled = (data - self.p25) / ((self.p75 - self.p25) + self.eps)
        return scaled * (maxim - minim) + minim
    
    def inverse_iqrnorm(self, data, minim=-1, maxim=1, **kwargs):
        scaled = ((data - minim) / (maxim - minim))
        return scaled * ((self.p75 - self.p25) + self.eps) + self.p25
    
    @property
    def iqrnorm(self): 
        return ( self._map_transform(self.forward_iqrnorm), 
                 self._map_transform(self.inverse_iqrnorm) )


    # Adaptively map data values to a given range to mitigate outliers
    def forward_adaptnorm(self, data, **kwargs): 
        if (self.min < 0) and (self.max > 0):
            scale = min(abs(self.min), self.max)
            kwargs['minim'] = self.min / scale
            kwargs['maxim'] = self.max / scale
        return self.forward_normalize(data, **kwargs)

    def inverse_adaptnorm(self, data, **kwargs): 
        if (self.min < 0) and (self.max > 0):
            scale = min(abs(self.min), self.max)
            kwargs['minim'] = self.min / scale
            kwargs['maxim'] = self.max / scale
        return self.inverse_normalize(data, **kwargs)

    @property
    def adaptnorm(self): 
        return ( self._map_transform(self.forward_adaptnorm), 
                 self._map_transform(self.inverse_adaptnorm) )


    # Transforms data into a gaussian (or uniform) distribution
    def forward_quantile(self, data, target='gaussian', **kwargs):
        targets = getattr(self, f'_{target}_targets')
        return self._quantile_transform(data, self._quantiles, targets)

    def inverse_quantile(self, data, target='gaussian', **kwargs):
        targets = getattr(self, f'_{target}_targets')
        return self._quantile_transform(data, targets, self._quantiles)

    @property
    def quantile(self):
        return ( self._map_transform(self.forward_quantile), 
                 self._map_transform(self.inverse_quantile) )


    # asinh scaling
    def forward_asinh(self, data, **kwargs):
        scale = (self.p75 - self.p25) / 1.349
        base = tf.math if isinstance(data, tf.Tensor) else np
        return base.asinh(data / (scale + self.eps))

    def inverse_asinh(self, data, **kwargs): 
        scale = (self.p75 - self.p25) / 1.349
        base = tf.math if isinstance(data, tf.Tensor) else np
        return (scale + self.eps) * base.sinh(data)
        
    @property
    def asinh(self): 
        return ( self._map_transform(self.forward_asinh), 
                 self._map_transform(self.inverse_asinh) )


    # Identity function
    def forward_identity(self, data, **kwargs): 
        return data   
        
    def inverse_identity(self, data, **kwargs): 
        return data
        
    @property
    def identity(self): 
        return ( self._map_transform(self.forward_identity), 
                 self._map_transform(self.inverse_identity) )

        
    # Used to create combinations of transform functions
    def forward_combination(self, data, transforms, **kwargs):
        """ Forward function only needs to apply transforms in order """
        # Use only the requested feature key, if one was given
        if kwargs.get('key', None) is not None: 
            self = self[kwargs['key']]

        for t in transforms:
            # Apply the current transform function to the data and stats
            func = self._get_transform(t, 0, **kwargs)
            self = self._transform_obj(func)
            data = func(data)
        return data

    def inverse_combination(self, data, transforms, **kwargs):
        """ Inverse function needs to apply transforms in reverse, as well
            as apply forward transforms in order to the statistics. """
        # Use only the requested feature key, if one was given
        if kwargs.get('key', None) is not None: 
            self = self[kwargs['key']]

        # First apply forward transforms to the stats, excluding the last 
        objs = [self]
        for t in transforms[:-1]:
            func = self._get_transform(t, 0, **kwargs)
            self = self._transform_obj(func)
            objs.append(self)

        # Then apply inverse transforms in reverse
        for t, self in zip(transforms[::-1], objs[::-1]):
            data = self._get_transform(t, 1, **kwargs)(data)
        return data

    def combination(self, transforms: list[Union[str, Callable]]):
        """ Create a forward (inverse) function that applies the given
            transforms in (reverse) order, while also applying each transform
            to the Transformer statistics to enable correct transforms """
        return ( self._map_transform(self.forward_combination, transforms=transforms), 
                 self._map_transform(self.inverse_combination, transforms=transforms) )


    # Internal functions
    # ==================
    def __getattr__(self, stat: str) -> Union[float, int, xr.DataArray]:
        """ Get a statistic for the current data, e.g. self.mean """
        stats = object.__getattribute__(self, 'stats')
        if stat == 'stats': return stats

        if stat in stats.statistics:
            value = stats.sel(statistics=stat)
            if value.size == 1: value = value.item()
            return value
        try:    return object.__getattribute__(self, stat)
        except: raise AttributeError(f'{self} has no attribute "{stat}"')


    def __getitem__(self, feature: str) -> 'Transform':
        """ Select a single feature from the stats and return a new object """
        if feature == '*':
            return self.copy()
        if 'features' not in (stats := self.stats).dims:
            stats = stats.expand_dims('features')
        return self.copy(statistics = stats.sel(features=feature))


    def _get_transform(self, transform, index: int = 0, **kwargs) -> Callable:
        """ Return the callable associated with the given transform """

        # Allow selecting transforms by name, and chaining them with commas 
        if isinstance(transform, str):

            # Get any named transform methods
            if hasattr(self, transform.strip()):
                transform = getattr(self, transform.strip())

            # Create a combination transform if multiple transforms are given
            elif ',' in transform:
                transform = self.combination([*transform.split(',')])

            # Parse the given transform keyword arguments to be applied
            elif '|' in transform:
                transform, *kwarg = transform.split('|')
                transform = getattr(self, transform.strip())
                for kv in kwarg:
                    k,v = kv.split('=')
                    try: v = float(v)
                    except: pass
                    kwargs[k] = v

        # Select the requested index in the (forward, inverse) pain
        if not isinstance(transform, str) and hasattr(transform, '__len__'):
            assert(len(transform)==2), f'Size must be exactly 2: {transform=}'
            transform = transform[index]
        
        assert(callable(transform)), f'Unknown transform: {transform=}'
        return partial(transform, **kwargs)


    def _map_transform(self, transform: Callable, **kwargs):
        """ Apply a transform to every value in the data dictionary """

        @wraps(transform)
        def wrapper(data, **kws):
            nonlocal transform, kwargs
            kws |= kwargs
            
            # Just apply the transform if data isn't a dictionary
            if not isinstance(data, dict):
                return transform(data, **kws)
            
            # If this transform is a bound method of a Transform object 
            if isinstance(getattr(transform, '__self__', None), Transform):
                 # Replace the object with a copy that only has stats for one key
                orig = transform.__self__
                func = transform.__func__.__get__
                copy = lambda k: func(orig[k], orig.__class__)
    
                # Create a new transform function that is unique for each feature
                transform = lambda v, key, **k: copy(key)(v, key=key, **k)
    
            return {k: transform(v, key=k, **kws) if k in self.stats.features
                    else v for k,v in data.items()}
        return wrapper


    def _process_feature(self, data, index: int, key: str, **kwargs):
        """Apply the transform defined in the dict that was given at init. 
        
        Parameters
        ----------
        data
            Data to transform; can be any type that the selected transform is
            able to utilize. 
        index : int
            Index of 0 indicates forward transform, and 1 indicates inverse.
        feature : str
            Feature name for the given data, which is used to select the 
            respective transform in the pre-defined dictionary.
        **kwargs
            Any other keyword arguments to be passed to the transform function.
        
        Returns
        -------
        T
            Returns the result of applying the feature transformation to the
            data, i.e. the same return type as the underlying transformation.
            This should be the same type as the data that was originally given.

        """
        
        feature = key if key in self.feature_transforms else '*'
        assert(feature in self.feature_transforms), f'Missing key: {key=}'
        try: 
            # Allow multiple transforms to be applied and returned as a list
            name_list = self.feature_transforms[feature].split('&')
            functions = [self._get_transform(n, index) for n in name_list]
            transform = [partial(f, key=feature, **kwargs) for f in functions]

            # For backwards compatibility, don't return a sole item as a list
            if len(transform) == 1:
                return transform[0](data)

            # When there is more than one transform to be applied:
            #   - Forward pass (index=0) applies each one to the data object
            #   - Inverse pass (index=1) applies respective inverse functions
            def process(features: list):
                """ Inverse needs to apply each respective transform """
                if len(features) == len(transform):
                    return [t(f) for t, f in zip(transform, features)]
                message = f'Expected {len(transform)=} items: {len(features)=}'
                raise Exception(message)

            # One-to-one correspondence between list items and transforms
            if (index == 1) and isinstance(data, list):
                return process(data)

            # Likewise, dicts must contain the feature and correctly sized list
            if (index == 1) and isinstance(data, dict):
                assert(feature in data), f'Missing {feature=}: {list(data)}'
                return {feature: process(data[feature])} 

            # Apply each transform to the data object during the forward pass,
            # as well as any inverse pass with a singular data object given
            return [t(data) for t in transform]

        except Exception as e: 
            tb = f'{e}\n{traceback.format_exc()}\ndata={str(data)[:100]}\n'
            raise Exception(f'{tb} Failed data transform for {key=} {index=}')


    def _infer_affine(self, prev, post, min_k: str, max_k: str, tol=1e-4):
        """ Infer affine transformation parameters """
        min_prev, max_prev = prev[min_k].values, prev[max_k].values
        min_post, max_post = post[min_k].values, post[max_k].values

        # Sanity check that the min/max values are reasonably spaced
        if np.mean(max_prev - min_prev) > tol:
            a = (max_post - min_post) / (max_prev - min_prev)
            b = min_post - a * min_prev
            return a, b


    def _transform_obj(self, transform: Callable, tol=1e-3) -> 'Transform':
        """ Attempt to update statistics to be consistent with transform """
        prev = self.stats.copy().to_dataset('statistics')
        post = transform(self.stats).to_dataset('statistics')
        keys = list(prev)

        # If transform is affine, transformed mid pt == mid pt of transformed
        trans_mid = transform((prev['p25'].values + prev['p75'].values)/2.)
        mid_trans = (post['p25'].values + post['p75'].values) / 2.
        use_tukey = np.any(np.mean(abs(trans_mid - mid_trans)) > tol)

        # Average the inferred parameters found using min/max and p25/p75
        if not use_tukey:
            a1b1 = self._infer_affine(prev, post, 'min', 'max', tol)
            a2b2 = self._infer_affine(prev, post, 'p25', 'p75', tol) or a1b1
            assert(a2b2 is not None), f'Summary update error: {prev=}\n{post=}'
            
            if np.all(np.abs(np.diff([a1b1 or a2b2, a2b2], axis=0))[0] < tol):
                a,b = np.mean([a1b1 or a2b2, a2b2], axis=0)
                est = {k: prev[k].values * abs(a) if k == 'std' else 
                          prev[k].values * a + b for k in keys}
            else: use_tukey = True

        # If transform fails affine checks, instead use Tukey's trimean
        if use_tukey:
            est = {k: post[k].values for k in keys} | {
                'median': post['median'].values,
                'mean'  : (post['p25'].values + post['p75'].values + 2*post['median'].values) / 4.,
                'std'   : abs(post['p75'].values - post['p25'].values) / 1.349,
            }

            # Transform flips the sign of the data, so percentiles are reversed
            if np.any(post['min'].values > post['max'].values):
                # While handling this scenario is easy enough for known stats:
                #   { 'min' : np.min([post['min'], post['max']], axis=0),
                #     'max' : np.max([post['min'], post['max']], axis=0),
                #     'p25' : np.min([post['p25'], post['p75']], axis=0),
                #     'p75' : np.max([post['p25'], post['p75']], axis=0) }
                #
                # it becomes much more complicated to handle universally, since
                # we would need to find the respective values on each side of 
                # the median to compare against (e.g. [p1,p99], [p6,p94], ...).
                # 
                # While it's unclear how to handle this for anything other than
                # percentiles, even for these it may not be possible if the 
                # respective value doesn't exist (e.g. p30 is available but p70
                # is not). Simply switching the key to its respective partner 
                # may have unintended side effects due to the available 
                # statistics changing.
                #
                # Due to these issues, sign flip transforms are not allowed. 
                raise Exception(f'{transform=} flips the sign of the data so'+
                        ' that min and max are swapped, which is not allowed')

        # Update the summary statistics and return a new Transform object
        # As tf.AutoGraph fails for Dataset.update, we update values directly
        keys = list(prev)
        prev = prev.to_array('statistics')
        prev.values = [est[k] for k in keys]
        return self.copy(statistics = prev)


    @cached_property
    def _quantiles(self):
        """ Return the quantiles available for this data """

        # Parse the p(Float) keys into their quantile and percentile values
        names = {
            'min'    : 'p0.5',
            'max'    : 'p99.5',
            'median' : 'p50',
        }
        parse = lambda k: (getattr(self, k), float(names.get(k, k)[1:]))
        valid = lambda k: (len(k) <= 3 and k[0] == 'p') or (k in names)
        order = lambda v: sorted(v, key=lambda qp: qp[1])

        # Gather the available quantiles and their respective percentiles
        quantiles, percentiles = map(np.array, zip( *order(
            [parse(k) for k in self.stats.statistics.values if valid(k)]) ))

        # If this is a single feature
        if 'features' not in self.stats.dims:

            # If the unique count is < 50% of the total number of bins
            if (len(np.unique(quantiles)) / len(quantiles)) < 0.5:

                # Gather the unique values, and include the min/max
                q, count = np.unique(quantiles, return_counts=True)
                if q[0] > self.min:    q = [self.min] + q
                if q[-1] < self.max:   q = q + [self.max]
                quantiles = np.array(q)

                # Construct new percentiles, placing duplicates in the middle 
                dup_ix = int(np.argmax(count))
                percentiles = []
                if dup_ix > 0:
                    lt_dup_vals = np.linspace(1, 50, dup_ix-1, endpoint=False)
                    percentiles += [0.5] + list(lt_dup_vals)

                # Place the most duplicated value at the 50th percentile
                n_gt = max((len(quantiles)-dup_ix)-2, 0)
                gt_dup_vals = np.linspace(99, 50, n_gt, endpoint=False)[::-1]
                percentiles += [50] + list(gt_dup_vals)

                # Add in the max (if it wasn't the duplicate)
                if dup_ix < (len(quantiles)-1):
                    percentiles += [99.5]

                percentiles = np.array(percentiles)
                assert(len(percentiles) == len(quantiles)), f'{dup_ix=} {percentiles=}'

        self._percentiles = percentiles.astype('float32')
        return quantiles.astype(f'float32')


    @cached_property
    def _gaussian_targets(self):
        """ Gaussian targets for a quantile transform """
        self._quantiles
        targets = norm.ppf(self._percentiles/100.)
        targets = targets / targets.max()
        return targets.astype('float32')


    @cached_property
    def _uniform_targets(self):
        """ Uniform targets in [0, 1] for a quantile transform """
        return np.linspace(0, 1, len(self._quantiles), dtype='float32')


    @cached_property
    def _uniform11_targets(self):
        """ Uniform targets in [-1, 1] for a quantile transform """
        return np.linspace(-1, 1, len(self._quantiles), dtype='float32')


    def _quantile_transform(self, x, quantiles, targets):
        """ Transform using the data quantiles and the target distribution """

        def _do_transform(x):
            x = tf.cast(x, tf.float32)
            
            # Clip input values to quantile bounds to avoid extrapolation
            x_clipped = tf.clip_by_value(x, quantiles[0], quantiles[-1])
    
            # Flatten input to [N] for easier processing
            x_flat = tf.reshape(x_clipped, [-1])  # shape [N]
            n_bins = tf.shape(quantiles)[0]
    
            # Find bin index for each value
            # Right bin: smallest i such that x <= quantiles[i]
            bin_idx = tf.searchsorted(quantiles, x_flat, side='right') - 1
            bin_idx = tf.clip_by_value(bin_idx, 0, n_bins - 2)  
    
            # Gather quantile bounds
            q_lo = tf.gather(quantiles, bin_idx)
            q_hi = tf.gather(quantiles, bin_idx + 1)
            t_lo = tf.gather(targets, bin_idx)
            t_hi = tf.gather(targets, bin_idx + 1)
    
            # Linear interpolation of new value
            slope = (t_hi - t_lo) / tf.maximum(q_hi - q_lo, 1e-4)
            value = t_lo + slope * (x_flat - q_lo)
    
            # Where q_low == q_high, just assign t_lo directly
            same_bin = tf.equal(q_lo, q_hi)
            t_interp = tf.where(same_bin, t_lo, value)
    
            # Reshape back to original input shape
            return tf.reshape(t_interp, tf.shape(x))

        # Handle non-tensorflow objects
        not_tensor = not any(k in str(type(x)) for k in ['tensorflow', 'keras'])
        if not_tensor:
            data = getattr(x, 'values', x)
            x_clipped = np.clip(data, quantiles[0], quantiles[-1])
            transform = np.interp(x_clipped, quantiles, targets)

            # Handle xarray objects
            if hasattr(x, 'loc'):
                x.values = transform
                return x
            return transform

            # Does not work if inside symbolic graph
            original, x = x, tf.constant(getattr(x, 'values', x))

        z = _do_transform(x)
        if not_tensor:
            z = getattr(z, 'numpy', lambda: z)()
            
            # Handle xarray objects
            if 'xarray' in str(type(original)):
                original.values = z
                return original
            return z
        return z
