"""

Copyright Parag K. Mital 2018. All rights reserved.
"""
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import dml
tfd = tf.contrib.distributions
tfb = tfd.bijectors
tfl = tf.layers


class RegressionHelper(tf.contrib.seq2seq.Helper):
    """Helper interface.    Helper instances are used by SamplingDecoder."""

    def __init__(self, batch_size, max_sequence_size, n_ch, embed=None):
        self._batch_size = batch_size
        self._max_sequence_size = max_sequence_size
        self._n_features = n_ch
        self._batch_size_tensor = tf.convert_to_tensor(
            batch_size, dtype=tf.int32, name="batch_size")
        if embed is not None:
            self._embed = embed
        else:
            self._embed = lambda x: x

    @property
    def batch_size(self):
        """Returns a scalar int32 tensor."""
        return self._batch_size_tensor

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._n_features

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        start_inputs = tf.fill([self._batch_size, self._n_features], 0.0)
        return (finished, self._embed(start_inputs))

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        del time, state
        return outputs

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        del sample_ids
        finished = tf.cond(
            tf.less(time, self._max_sequence_size), lambda: False, lambda: True)
        del time
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: tf.zeros_like(outputs),
            lambda: outputs)
        return (finished, self._embed(next_inputs), state)


class MDNRegressionHelper(tf.contrib.seq2seq.Helper):
    """Helper interface.    Helper instances are used by SamplingDecoder."""

    def __init__(self,
                 batch_size,
                 max_sequence_size,
                 n_ch,
                 n_mixtures,
                 embed=None):
        self._batch_size = batch_size
        self._max_sequence_size = max_sequence_size
        self._n_features = n_ch
        self._n_mixtures = n_mixtures
        self._batch_size_tensor = tf.convert_to_tensor(
            batch_size, dtype=tf.int32, name="batch_size")
        if embed is not None:
            self._embed = embed
        else:
            self._embed = lambda x: x

    @property
    def batch_size(self):
        """Returns a scalar int32 tensor."""
        return self._batch_size_tensor

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._n_features

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        start_inputs = tf.fill([self._batch_size, self._n_features], 0.0)
        return (finished, self._embed(start_inputs))

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        del time, state
        # return outputs
        with tf.variable_scope('mdn'):
            means = tf.reshape(
                tf.slice(
                    outputs, [0, 0],
                    [self._batch_size, self._n_features * self._n_mixtures]),
                [self._batch_size, self._n_features, self._n_mixtures],
                name='means')
            sigmas = tf.minimum(10000.0, tf.maximum(1e-1, tf.nn.softplus(
                    tf.reshape(
                        tf.slice(
                            outputs, [0, self._n_features * self._n_mixtures], [
                                self._batch_size,
                                self._n_features * self._n_mixtures
                            ],
                            name='sigmas_pre_norm'),
                        [self._batch_size, self._n_features, self._n_mixtures
                            ]))))
            weights = tf.nn.softmax(
                tf.reshape(
                    tf.slice(
                        outputs, [0, 2 * self._n_features * self._n_mixtures],
                        [self._batch_size, self._n_mixtures],
                        name='weights_pre_norm'),
                    [self._batch_size, self._n_mixtures]),
                name='weights')
            components = []
            for gauss_i in range(self._n_mixtures):
                mean_i = means[:, :, gauss_i]
                sigma_i = sigmas[:, :, gauss_i]
                components.append(
                    tfd.MultivariateNormalDiag(loc=mean_i, scale_diag=sigma_i))
            gauss = tfd.Mixture(
                cat=tfd.Categorical(probs=weights), components=components)
            sample = gauss.sample()
        return sample

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        finished = tf.cond(
            tf.less(time, self._max_sequence_size), lambda: False, lambda: True)
        del time
        del outputs
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: tf.zeros_like(sample_ids),
            lambda: sample_ids)
        del sample_ids
        return (finished, self._embed(next_inputs), state)


class DMLRegressionHelper(tf.contrib.seq2seq.Helper):
    """Helper interface.    Helper instances are used by SamplingDecoder."""

    def __init__(self,
                 batch_size,
                 max_sequence_size,
                 n_ch,
                 n_mixtures,
                 embed=None):
        self._batch_size = batch_size
        self._max_sequence_size = max_sequence_size
        self._n_features = n_ch
        self._n_mixtures = n_mixtures
        self._batch_size_tensor = tf.convert_to_tensor(
            batch_size, dtype=tf.int32, name="batch_size")
        if embed is not None:
            self._embed = embed
        else:
            self._embed = lambda x: x

    @property
    def batch_size(self):
        """Returns a scalar int32 tensor."""
        return self._batch_size_tensor

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._n_features

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        start_inputs = tf.fill([self._batch_size, self._n_features], 0.0)
        return (finished, self._embed(start_inputs))

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        del time, state
        # return outputs
        with tf.variable_scope('dml'):
            reshaped = tf.reshape(
                outputs,
                [self._batch_size, self._n_features, self._n_mixtures * 3])
            loc, unconstrained_scale, logits = tf.split(reshaped,
                                                        num_or_size_splits=3,
                                                        axis=-1)
            loc = tf.minimum(tf.nn.softplus(loc), 2.0 ** 16 - 1.0)
            scale = tf.minimum(14.0, tf.maximum(1e-8, tf.nn.softplus(unconstrained_scale)))
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.,
                high=2**16 - 1.)
            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)
            sample = tf.minimum(1.0, tf.maximum(-1.0, mixture_dist.sample() / (2**16 - 1.0)))
        return sample

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        finished = tf.cond(
            tf.less(time, self._max_sequence_size), lambda: False, lambda: True)
        del time
        del outputs
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: tf.zeros_like(sample_ids),
            lambda: sample_ids)
        del sample_ids
        return (finished, self._embed(next_inputs), state)
