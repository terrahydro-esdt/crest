import collections
import tensorflow as tf
import numpy as np
from scipy import stats
import traceback

class Metrics(object):

    def __init__(self):
        self.handlers = collections.defaultdict(set)

        # Custom metrics
        self.register('unbiased_rmse', self.unbiased_rmse)
        self.register('fluctuation_complexity', self.fluctuation_complexity)
        self.register('metric_entropy', self.metric_entropy)
        self.register('relative_error', self.relative_error)
        self.register('triple_collocation_error', self.triple_collocation_error)
        self.register('nse', self.nse)
        self.register('nse_log', self.nse_log)
        self.register('mse', self.mse)
        self.register('kge', self.kge)
        self.register('lkg', self.lkge)
        self.register('pearson', self.pearsonr)
        self.register('alpha_nse', self.alpha_nse)
        self.register('beta_nse', self.beta_nse)

    def get_callbacks(self, metrics):
        m_callbacks = []
        try:
            if (not isinstance(metrics, list)):
                metrics = [metrics]

            for m in metrics:
                if (isinstance(m, str)):
                    keras_avail = tf.keras.metrics.get(m) 
                    if ((not keras_avail == m) or callable(keras_avail)):
                        m_callbacks.append(keras_avail)
                        continue

                    handler_avail = self.get_handler(m)
                    if (not handler_avail is None):
                        m_callbacks.append(handler_avail)
                        continue
                    
                    raise Exception('Could not get metric callback')
                
                elif (callable(m)):
                    m_callbacks.append(m)

        except:
            print(traceback.format_exc())
            raise Exception('Could not get all metric callbacks')
        finally:
            return m_callbacks

    def register(self, event, callback):
        self.handlers[event].add(callback)

    def get_handler(self, event):
        for handler in self.handlers.get(event, []):
            return handler
    
    @property
    def all(self):
        return self.handlers.keys()
        
    def unbiased_rmse(self, y_true, y_pred):
        return (tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(y_true, y_pred), 2)))).numpy()
    
    # TODO: This needs to be implemented
    def fluctuation_complexity(self, y_true, y_pred):
        return

        import numpy as np
        import pandas as pd 
        import scipy.stats as stats

        log_returns_true = np.log(y_true).diff().dropna()
        log_returns_pred = np.log(y_pred).diff().dropna()

        time_scales_true = np.arange(10, len(log_returns_true), 10)
        rolling_max_true = log_returns_true.rolling(time_scales_true).max()
        rolling_min_true = log_returns_true.rolling(time_scales_true).min()
        range_true = rolling_max_true - rolling_min_true

        avg_range_true = tf.reduce_mean(range_true)
        log_avg_range_true = np.log(avg_range_true)
        log_time_scales = np.log(time_scales_true)

        slope_true, intercept_true, r_value_true, p_value_true, std_err_true = stats.linregress(log_time_scales, log_avg_range_true)

        hurst_exponent_true = slope_true
        fluctuation_complexity_true = 0.5 * (hurst_exponent_true + 2)

        return fluctuation_complexity_true

    def metric_entropy(self, y_true, y_pred):
        import scipy.stats as stats
        norm_entropy_true = stats.entropy(y_true) / tf.size(y_true)
        norm_entropy_pred = stats.entropy(y_pred) / tf.size(y_pred)

        return (norm_entropy_true, norm_entropy_pred)
        
    def relative_error(self, y_true, y_pred):
        result = tf.math.divide(tf.math.subtract(y_pred, y_true), y_true)
        result = tf.where(tf.math.is_nan(result), tf.zeros_like(result), result)
        return tf.reduce_mean(result)

    # TODO: This needs to be implemented
    def triple_collocation_error(self, y_true, y_pred):
        return

    def nse(self, y_true, y_pred):
        denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        numerator = tf.reduce_sum(tf.square(y_pred - y_true))

        return float(1 - (numerator / denominator))

    def nse_log(self, y_true, y_pred):
        return tf.math.log(self.nse(y_true, y_pred))

    def mse(self, y_true, y_pred):
        return tf.reduce_mean(tf.pow(tf.subtract(y_true, y_pred), 2))
    
    def kge(self, y_true, y_pred, weights: [float] = [1., 1., 1.]):
        if len(y_true) < 2:
            return np.nan

        r = tf.subtract(self.pearsonr(y_true, y_pred), 1)
        alpha = tf.subtract(tf.math.reduce_std(y_pred) / tf.math.reduce_std(y_true), 1)
        beta = tf.subtract(tf.reduce_mean(y_pred) / tf.reduce_mean(y_true), 1)

        val_1 = tf.cast(tf.multiply(weights[0], tf.square(r)), 'float')
        val_2 = tf.cast(tf.multiply(weights[1], tf.square(alpha)), 'float')
        val_3 = tf.cast(tf.multiply(weights[2], tf.square(beta)), 'float')

        sum = float(tf.add_n([val_1, val_2, val_3]).numpy())
        value = float(tf.subtract(1, tf.sqrt(sum)))

        return value

    def lkge(self, y_true, y_pred):
        return tf.math.log(self.kge(y_true, y_pred))        

    def pearsonr(self, y_true, y_pred):
        r, _ = stats.pearsonr(y_true, y_pred)

        return float(r)
    
    def alpha_nse(self, y_true, y_pred):
        return float(tf.math.reduce_std(y_pred) / tf.math.reduce_std(y_true))

    def beta_nse(self, y_true, y_pred):
        return float((tf.reduce_mean(y_pred) - tf.reduce_mean(y_true)) / tf.math.reduce_std(y_true))
