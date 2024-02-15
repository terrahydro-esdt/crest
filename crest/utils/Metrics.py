import collections
import tensorflow as tf
import numpy as np
from scipy import stats
import jsonpickle


class Metrics(object):
    """
    The Metrics class is a wrapper around custom metrics that can be 
    used in the CREST framework. It also manages the keras metrics used by the 
    Model class. It allows for easy saving and loading of custom metrics via 
    JSON serialization.
    """

    def __init__(self):
        """
        Initializes the Metrics class with a dictionary of handlers for
        custom metrics.
        """
        self.handlers = collections.defaultdict(set)

        # Custom metrics
        self.register('unbiased_rmse', self.unbiased_rmse)
        self.register('fluctuation_complexity', self.fluctuation_complexity)
        self.register('metric_entropy', self.metric_entropy)
        self.register('relative_error', self.relative_error)
        self.register('triple_collocation_error',
                      self.triple_collocation_error)
        self.register('nse', self.nse)
        self.register('nse_log', self.nse_log)
        self.register('mse', self.mse)
        self.register('kge', self.kge)
        self.register('lkg', self.lkge)
        self.register('pearson', self.pearsonr)
        self.register('alpha_nse', self.alpha_nse)
        self.register('beta_nse', self.beta_nse)

        self.defaults = list(self.handlers.keys())

    def get_callbacks(self, metrics):
        """
        Get the callbacks for the given metrics. If the metric is a string,
        it will check if it is a keras metric. If it is not, it will check if it
        is a custom metric. If the metric is a callable, it will be added to the
        list of callbacks.

        Parameters
        ----------
            metrics: A list of metrics for which callbacks are required
        """
        m_callbacks = []
        try:
            # If metrics is not a list, make it a list
            if not isinstance(metrics, list):
                metrics = [metrics]

            # For each metric, check if it is a keras metric or a custom metric
            for m in metrics:

                # If the metric is a string
                if isinstance(m, str):
                    keras_avail = tf.keras.metrics.get(m)
                    handler_avail = self.get_handler(m)

                    # If the metric is a keras metric, add it to the list of callbacks
                    if (not keras_avail == m) or callable(keras_avail):
                        m_callbacks.append(keras_avail)
                    elif not handler_avail is None:
                        m_callbacks.append(handler_avail)

                # If the metric is a callable, add it to the list of callbacks
                elif callable(m):
                    self.register(m.__class__.__name__, m)
                    m_callbacks.append(m)

        finally:
            return m_callbacks

    def register(self, event, callback):
        """
        Register a custom metric with the given event and callback.

        Parameters
        ----------
            event: The event for which the callback is to be registered
            callback: The callback to be registered
        """
        self.handlers[event].add(callback)

    def get_handler(self, event):
        """
        Get the handler for the given event.

        Parameters
        ----------
            event: The event for which the handler is required

        Returns
        -------
            The handler for the given event
        """
        for handler in self.handlers.get(event, []):
            return handler

    @property
    def all(self):
        """
        Property to get all the registered metrics.
        """
        return list(self.handlers.keys())

    @property
    def customs(self):
        """
        Property to get all the custom metrics.
        """
        return [k for k in self.all if not k in self.defaults]

    def unbiased_rmse(self, y_true, y_pred):
        """
        Unbiased Root Mean Squared Error (RMSE) metric.
        The RMSE is a measure of the differences between values predicted 
        by a model or an estimator and the values actually observed.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The RMSE value
        """
        return (tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(y_true, y_pred), 2)))).numpy()

    # TODO: This needs to be implemented
    def fluctuation_complexity(self, y_true, y_pred):
        """
        Fluctuation Complexity metric.
        The Fluctuation Complexity is a measure of the complexity of a time series.
        It is calculated as the average of the Hurst exponent and 2.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The Fluctuation Complexity value
        """
        return

        # throw NotImplementedError('Fluctuation Complexity is not implemented yet')

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

        slope_true, intercept_true, r_value_true, p_value_true, std_err_true = stats.linregress(
            log_time_scales, log_avg_range_true)

        hurst_exponent_true = slope_true
        fluctuation_complexity_true = 0.5 * (hurst_exponent_true + 2)

        return fluctuation_complexity_true

    def metric_entropy(self, y_true, y_pred):
        """
        Metric Entropy metric.
        The Metric Entropy is a measure of the entropy of a time series.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The Metric Entropy value
        """
        import scipy.stats as stats
        norm_entropy_true = stats.entropy(y_true) / tf.size(y_true)
        norm_entropy_pred = stats.entropy(y_pred) / tf.size(y_pred)

        return norm_entropy_true, norm_entropy_pred

    def relative_error(self, y_true, y_pred):
        """
        Relative Error metric.
        The Relative Error is a measure of the error in a model's predictions.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The Relative Error value
        """
        result = tf.math.divide(tf.math.subtract(y_pred, y_true), y_true)
        result = tf.where(tf.math.is_nan(result),
                          tf.zeros_like(result), result)
        return tf.reduce_mean(result)

    # TODO: This needs to be implemented
    def triple_collocation_error(self, y_true, y_pred):
        """
        Triple Collocation Error metric.
        The Triple Collocation Error is a measure of the error in three
        independent measurements of the same quantity.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The Triple Collocation Error value
        """
        return

        # throw NotImplementedError('Triple Collocation Error is not implemented yet')

    def nse(self, y_true, y_pred):
        """
        Nash-Sutcliffe Efficiency (NSE) metric.
        The Nash-Sutcliffe Efficiency is a measure of the accuracy of a model
        in predicting values.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The NSE value
        """
        denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        numerator = tf.reduce_sum(tf.square(y_pred - y_true))

        return float(1 - (numerator / denominator))

    def nse_log(self, y_true, y_pred):
        """
        Logorithimic Nash-Sutcliffe Efficiency (lNSE) metric.
        The log NSE is a measure of the accuracy of a model in predicting values.
        The log NSE increases the sensitivity to low flows.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The NSE value in log space
        """

        log_obs = tf.math.log(y_true)
        log_sim = tf.math.log(y_pred)

        return self.nse(log_obs, log_sim)

    def mse(self, y_true, y_pred):
        """
        Mean Squared Error (MSE) metric.
        The MSE is a measure of the differences between values predicted
        by a model or an estimator and the values actually observed.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Returns
        -------
            The MSE value
        """
        return tf.reduce_mean(tf.pow(tf.subtract(y_true, y_pred), 2))

    def kge(self, y_true, y_pred, weights: [float] = [1., 1., 1.]):
        """
        Kling-Gupta Efficiency (KGE) metric.
        The KGE is a measure of the accuracy of a model in predicting values.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values
            weights: The weights for the KGE components

        Returns
        -------
            The KGE value
        """
        if len(y_true) < 2:
            return np.nan

        r = tf.subtract(self.pearsonr(y_true, y_pred), 1)
        alpha = tf.subtract(tf.math.reduce_std(y_pred) /
                            tf.math.reduce_std(y_true), 1)
        beta = tf.subtract(tf.reduce_mean(y_pred) / tf.reduce_mean(y_true), 1)

        val_1 = tf.cast(tf.multiply(weights[0], tf.square(r)), 'float')
        val_2 = tf.cast(tf.multiply(weights[1], tf.square(alpha)), 'float')
        val_3 = tf.cast(tf.multiply(weights[2], tf.square(beta)), 'float')

        sum = float(tf.add_n([val_1, val_2, val_3]).numpy())
        value = float(tf.subtract(1, tf.sqrt(sum)))

        return value

    def lkge(self, y_true, y_pred):
        """
        Logarithmic Kling-Gupta Efficiency (KGE) metric.
        The KGE is a measure of the accuracy of a model in predicting values.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Return
        ------
            The KGE value in log space
        """
        return tf.math.log(self.kge(y_true, y_pred))

    def pearsonr(self, y_true, y_pred):
        """
        Pearson correlation coefficient metric.
        The Pearson correlation coefficient is a measure of the linear correlation
        between two variables.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Return
        ------
            The Pearson correlation coefficient value
        """
        r, _ = stats.pearsonr(y_true, y_pred)

        return float(r)

    def alpha_nse(self, y_true, y_pred):
        """
        Alpha Nash-Sutcliffe Efficiency (NSE) metric.
        The alpha NSE is a fraction of the standard deviation of the predicted
        values to the standard deviation of the true values.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Return
        ------
            The Alpha-NSE value
        """
        return float(tf.math.reduce_std(y_pred) / tf.math.reduce_std(y_true))

    def beta_nse(self, y_true, y_pred):
        """
        Beta Nash-Sutcliffe Efficiency (NSE) metric.
        The beta NSE is the difference between the mean of the predicted values
        and the mean of the true values, divided by the standard deviation of the
        true values.

        Parameters
        ----------
            y_true: The true values
            y_pred: The predicted values

        Return
        ------
            The Beta-NSE value
        """
        return float((tf.reduce_mean(y_pred) - tf.reduce_mean(y_true)) / tf.math.reduce_std(y_true))

    def to_json(self):
        """
        The JSON representation of the Metrics class.

        Return
        ------
            str: The JSON representation of the Metrics class
        """
        registered_metrics = list(self.handlers.keys())

        saved_dict = {}
        for metric in registered_metrics:
            saved_dict[metric] = self.get_handler(metric)

        saved_dict = jsonpickle.encode(saved_dict)
        return saved_dict

    @staticmethod
    def from_json(saved_dict):
        """
        Loads the Metrics class from a JSON representation.

        Pameter
        -------
            saved_dict: The JSON representation of the Metrics class

        Return
        ------
            Metrics: The Metrics class loaded from the JSON representation
        """
        saved_dict = jsonpickle.decode(saved_dict)
        m = Metrics()
        metrics = list(saved_dict.keys())

        for metric in metrics:
            if not metric in m.all:
                m.register(metric, saved_dict[metric])

        return m
