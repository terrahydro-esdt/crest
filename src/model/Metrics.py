
import collections
import tensorflow as tf

class Metrics(object):

    def __init__(self, metrics):
        self.handlers = collections.defaultdict(set)

        self.register('unbiased_rmse', self.unbiased_rmse)
        self.register('fluctuation_complexity', self.fluctuation_complexity)
        self.register('metric_entropy', self.metric_entropy)
        self.register('relative_error', self.relative_error)
        self.register('triple_collocation_error', self.triple_collocation_error)
        self.register('nse', self.nse)
        self.register('nse_log', self.nse_log)
        self.register('mse', self.mse)
        self.register('kg', self.kge)
        self.register('lkg', self.lkge)
        self.register('pearson', self.pearsonr)
        self.register('alpha_nse', self.alpha_nse)
        self.register('beta_nse', self.beta_nse)

    def crest_metrics(self, metrics):

        if all(isinstance(item, str) for item in metrics):
            mapped_metrics = []
            return [mapped_metrics.append(self.get_handler(metric)) for metric in metrics]
        else:
            raise Exception('Metrics must be a list of strings')

    def register(self, event, callback):
        self.handlers[event].add(callback)

    def get_handler(self, event):
        for handler in self.handlers.get(event, []):
            return handler
        
    def unbiased_rmse(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(y_true, y_pred), 2)))
    
    def fluctuation_complexity(self, y_true, y_pred):
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
        return tf.divide(tf.subtract(y_pred, y_true), y_true)

    def triple_collocation_error(self, **kwargs):
        pass

    def nse(self, y_true, y_pred):
        denominator = tf.reduce_sum(tf.pow(tf.subtract(y_true, tf.reduce_mean(y_pred)), 2))
        numerator = tf.reduce_sum(tf.pow(tf.subtract(y_true, y_true), 2))

        return float(1 - (numerator / denominator))

    # @TODO: Double check this
    def nse_log(self, y_true, y_pred):
        return tf.math.log(self.nse(y_true, y_pred))

    def mse(self, y_true, y_pred):
        return tf.reduce_mean(tf.pow(tf.subtract(y_true, y_pred), 2))

    # @TODO: Implement weights?
    def kge(self, y_true, y_pred):
        r = self.pearsonr(y_true, y_pred)
        alpha = self.alpha_nse(y_true, y_pred)
        beta = self.beta_nse(y_true, y_pred)

        kge = 1 - tf.sqrt(tf.pow(r - 1, 2) + tf.pow(alpha - 1, 2) + tf.pow(beta - 1, 2))
        return kge

    # @TODO: Double check this
    def lkge(self, y_true, y_pred):
        return tf.math.log(self.kge(y_true, y_pred))        

    def pearsonr(self, y_true, y_pred):
        epsilon = 10e-5
        mx = tf.keras.metrics.Mean(y_true)
        my = tf.keras.metrics.Mean(y_pred)

        xm = tf.math.subtract(y_true, mx)
        ym = tf.math.subtract(y_pred, my)
        
        r_num = tf.reduce_sum(tf.multiply(xm, ym))
        
        x_square_sum = tf.reduce_sum(tf.square(xm))
        y_square_sum = tf.reduce_sum(tf.square(ym))
        
        r_den = tf.sqrt(tf.multiply(x_square_sum, y_square_sum))
        mean_r = tf.keras.metrics.Mean(r_num / (r_den + epsilon))
        
        return mean_r

    def alpha_nse(self, y_true, y_pred):
        return float(tf.math.reduce_std(y_pred) / tf.math.reduce_std(y_true))

    def beta_nse(self, y_true, y_pred):
        return float((tf.reduce_mean(y_pred) - tf.reduce_mean(y_true)) / tf.std(y_true))