"""
Scheduled Sampling
Anneal input of decoder
conv embed

Weight norm
l2 regularization
initializers
Check batch generator / shuffling behavior for tfrecords

Copyright Parag K. Mital 2018. All rights reserved.
"""
import os
import librosa
import numpy as np
from scipy.signal import hann
from functools import partial
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
from decoders import RegressionHelper, MDNRegressionHelper, DMLRegressionHelper
import dml
from tensorflow.contrib.rnn import LSTMStateTuple
tfd = tf.contrib.distributions
tfb = tfd.bijectors
tfl = tf.layers

"""
This module is based on an implementation from:
    http://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow
"""


class BasicConvLSTMCell:

    def __init__(self,
                 shape,
                 filter_size,
                 num_features,
                 forget_bias=1.0,
                 state_is_tuple=False,
                 activation=tf.nn.tanh):
        """
          shape: int tuple, height and width of the cell
          filter_size: int tuple, height and width of the filter
          num_features: int, depth of the cell
          forget_bias: float, forget gates bias
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: activation function of the inner states
        """
        self.shape = shape
        self.height, self.width = self.shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._num_units = self.num_features * self.height * self.width
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=1, num_or_size_splits=2, value=state)

            batch_size = tf.shape(inputs)[0]
            inputs = tf.reshape(inputs,
                                [batch_size, self.height, self.width, 1])
            c = tf.reshape(
                c, [batch_size, self.height, self.width, self.num_features])
            h = tf.reshape(
                h, [batch_size, self.height, self.width, self.num_features])

            concat = _conv_linear([inputs, h], self.filter_size,
                                  self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) +
                     tf.nn.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            new_h = tf.reshape(new_h, [batch_size, self._num_units])
            new_c = tf.reshape(new_c, [batch_size, self._num_units])

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=1, values=[new_c, new_h])

            return new_h, new_state

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """
        return tf.zeros([batch_size, self._num_units * 2])


def _conv_linear(args,
                 filter_size,
                 num_features,
                 bias,
                 bias_start=0.0,
                 scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError(
                "Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError(
                "Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [
                filter_size[0], filter_size[1], total_arg_size_depth,
                num_features
            ],
            dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(
                args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(
                tf.concat(axis=3, values=args),
                matrix,
                strides=[1, 1, 1, 1],
                padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term


def compute_inputs(x, freqs, n_fft, n_frames, input_features, norm=False):
    if norm:
        norm_fn = instance_norm
    else:

        def norm_fn(x):
            return x

    freqs_tf = tf.constant(freqs, name="freqs", dtype='float32')
    inputs = {}
    with tf.variable_scope('real'):
        inputs['real'] = norm_fn(
            tf.reshape(
                tf.matmul(x, tf.cos(freqs_tf)), [1, 1, n_frames, n_fft // 2]))
    with tf.variable_scope('imag'):
        inputs['imag'] = norm_fn(
            tf.reshape(
                tf.matmul(x, tf.sin(freqs_tf)), [1, 1, n_frames, n_fft // 2]))
    with tf.variable_scope('mags'):
        inputs['mags'] = norm_fn(
            tf.reshape(
                tf.sqrt(
                    tf.maximum(1e-15, inputs['real'] * inputs['real'] +
                               inputs['imag'] * inputs['imag'])),
                [1, 1, n_frames, n_fft // 2]))
    with tf.variable_scope('phase'):
        inputs['phase'] = norm_fn(tf.atan2(inputs['imag'], inputs['real']))
    with tf.variable_scope('unwrapped'):
        inputs['unwrapped'] = tf.py_func(unwrap, [inputs['phase']], tf.float32)
    with tf.variable_scope('unwrapped_difference'):
        inputs['unwrapped_difference'] = (tf.slice(inputs['unwrapped'], [
            0, 0, 0, 1
        ], [-1, -1, -1, n_fft // 2 - 1]) - tf.slice(
            inputs['unwrapped'], [0, 0, 0, 0], [-1, -1, -1, n_fft // 2 - 1]))
    if 'unwrapped_difference' in input_features:
        for k, v in input_features:
            if k is not 'unwrapped_difference':
                inputs[k] = tf.slice(v, [0, 0, 0, 0],
                                     [-1, -1, -1, n_fft // 2 - 1])
    net = tf.concat([inputs[i] for i in input_features], 1)
    return inputs, net


def _vqembed(z_e, K=128, D=512):
    # Tile to K, then transpose
    z_e_tiled = tf.transpose(tf.tile(z_e, [1, 1, 1, K]), [0, 1, 3, 2])

    # Build VQ
    with tf.variable_scope('embed'):
        embeds = tf.get_variable(
            'embedding', [K, D],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.02))
    embeds_tiled = tf.reshape(embeds, [1, 1, K, D])
    dist = tf.norm(z_e_tiled - embeds_tiled, axis=-1)
    k = tf.argmin(dist, axis=-1)
    z_q = tf.gather(embeds, k)
    return z_q


def _create_embedding(x, embed_size, embed_matrix=None):
    batch_size, sequence_length, n_input = x.shape.as_list()
    # Creating an embedding matrix if one isn't given
    if embed_matrix is None:
        embed_matrix = tf.get_variable(
            name='embed_matrix',
            shape=[n_input, embed_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    embed = tf.reshape(
        tf.matmul(
            tf.reshape(x, [batch_size * sequence_length, n_input]),
            embed_matrix), [batch_size, sequence_length, embed_size])
    return embed, embed_matrix


def _create_rnn_cell(
        n_neurons,
        n_layers,
        keep_prob,
        wrapper='highway',
        cell_type='intersection',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        **cell_type_kwargs):
    if wrapper == 'highway':
        wrapper = rnn.HighwayWrapper
    elif wrapper == 'residual':
        wrapper = rnn.ResidualWrapper
    elif wrapper == 'dropout':
        wrapper = partial(rnn.DropoutWrapper, output_keep_prob=keep_prob)
    elif wrapper == 'none' or wrapper is None:
        wrapper = lambda x: x
    else:
        raise NotImplementedError("Not implemented wrapper")

    if cell_type == 'layer-norm-lstm':
        cell_type = partial(
            rnn.LayerNormBasicLSTMCell, dropout_keep_prob=keep_prob)
    elif cell_type == 'lstm':
        cell_type = rnn.BasicLSTMCell
    elif cell_type == 'conv-lstm':
        cell_type = BasicConvLSTMCell
    elif cell_type == 'group-lstm':
        cell_type = rnn.GLSTMCell
    elif cell_type == 'gru':
        cell_type = partial(rnn.GRUCell, kernel_initializer=kernel_initializer)
    elif cell_type == 'intersection':
        cell_type = partial(
            rnn.IntersectionRNNCell, initializer=kernel_initializer)
    elif cell_type == 'phased':
        cell_type = rnn.PhasedLSTMCell
    else:
        raise NotImplementedError("Not implemented cell type")

    cell_fw = wrapper(cell_type(num_units=n_neurons))
    # Build deeper recurrent net if using more than 1 layer
    if n_layers > 1:
        cells = [cell_fw]
        for layer_i in range(1, n_layers):
            with tf.variable_scope('{}'.format(layer_i)):
                cell_fw = wrapper(cell_type(num_units=n_neurons))
                cells.append(cell_fw)
        cell_fw = rnn.MultiRNNCell(cells)
    return cell_fw


def _create_encoder(source,
                    lengths,
                    batch_size,
                    n_neurons,
                    n_layers,
                    keep_prob,
                    wrapper,
                    cell_type,
                    input_layer,
                    use_bi=False):
    source_embed = input_layer(source)

    # Create the RNN Cells for encoder
    with tf.variable_scope('forward'):
        cell_fw = _create_rnn_cell(n_neurons, n_layers, keep_prob, wrapper,
                                   cell_type)

    # Now hookup the cells to the input
    # [batch_size, max_time, embed_size]
    if use_bi:
        # Create the internal multi-layer cell for the backward RNN.
        with tf.variable_scope('backward'):
            cell_bw = _create_rnn_cell(n_neurons, n_layers, keep_prob, wrapper,
                                       cell_type)

        (outputs_fw, output_bw), (final_state_fw, final_state_bw) = \
            tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=source_embed,
                sequence_length=lengths,
                time_major=False,
                dtype=tf.float32)
    else:
        outputs_fw, final_state_fw = \
            tf.nn.dynamic_rnn(
                cell=cell_fw,
                inputs=source_embed,
                sequence_length=lengths,
                time_major=False,
                dtype=tf.float32)

    return outputs_fw, final_state_fw


def _create_decoder(n_neurons,
                    n_layers,
                    keep_prob,
                    dec_in_keep_prob,
                    batch_size,
                    encoder_outputs,
                    encoder_state,
                    encoder_lengths,
                    decoding_inputs,
                    decoding_lengths,
                    synth_length,
                    n_ch,
                    scope,
                    max_sequence_size,
                    n_mixtures,
                    wrapper,
                    cell_type,
                    input_layer,
                    use_attention=False,
                    n_code=256,
                    n_bijectors=10,
                    n_made_layers=3,
                    n_made_code_factor=2,
                    model='mdn',
                    variational='MAF'):
    if model == 'mdn':
        n_outputs = n_ch * n_mixtures + n_ch * n_mixtures + n_mixtures
    elif model == 'dml':
        n_outputs = n_mixtures * n_ch * 3
    else:
        n_outputs = n_ch
    output_layer = tfl.Dense(n_outputs, name='output_projection')

    with tf.variable_scope('forward'):
        cells = _create_rnn_cell(n_neurons, n_layers, keep_prob, wrapper,
                                 cell_type)

    if use_attention:
        attn_mech = tf.contrib.seq2seq.LuongAttention(
            cells.output_size, encoder_outputs, encoder_lengths, scale=True)
        cells = tf.contrib.seq2seq.AttentionWrapper(
            cell=cells,
            attention_mechanism=attn_mech,
            attention_layer_size=cells.output_size,
            alignment_history=False)
        initial_state = cells.zero_state(
            dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)
    else:
        initial_state = encoder_state

    losses = tf.Variable(0.0, trainable=False)
    if variational is not None:
        if use_attention:
            states = tf.concat(initial_state.cell_state, -1)
            # TODO: How do I proprely combine attention and recreate the cells?
            # states = tf.concat([
            #     tf.concat(initial_state.cell_state, -1),
            #     initial_state.attention_state], -1)
            raise NotImplementedError(
                'not working yet... cannot combine attention and variational layers'
            )
        else:
            states = tf.concat(initial_state, -1)
        state_size = int(states.shape[-1])

        z_mus = tfl.dense(states, n_code, name='mus')
        z_log_sigmas = tf.minimum(5.0,
                                  tf.maximum(1e-3,
                                             tf.nn.softplus(
                                                 tfl.dense(
                                                     states,
                                                     n_code,
                                                     name='log_sigmas'))))
        if isinstance(initial_state[0], rnn.LSTMStateTuple):
            z_mus = tf.reshape(z_mus, (2 * batch_size, n_code))
            z_log_sigmas = tf.reshape(z_log_sigmas, (2 * batch_size, n_code))
        if variational == 'VAE':
            prior = tfd.MultivariateNormalDiag(
                loc=z_mus, scale_diag=tf.exp(z_log_sigmas))
            z_sample = tf.concat((states, prior.sample()), -1)
            # z_sample = prior.sample()
            losses -= tf.reduce_mean(prior.log_prob(z_mus))
        elif variational == 'IAF':
            prior = tfd.MultivariateNormalDiag(
                loc=z_mus, scale_diag=tf.exp(z_log_sigmas))
            n_made_code = max(n_code + 1, n_made_code_factor * n_code)
            bijectors = []
            for i in range(n_bijectors):
                bijector = tfb.Invert(
                    tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=tfb.
                        masked_autoregressive_default_template(
                            hidden_layers=[n_made_code] * n_made_layers)))
                bijectors.append(
                    tfb.Permute(permutation=list(range(n_code - 1, -1, -1))))
            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
            flow = tfd.TransformedDistribution(
                distribution=prior, bijector=flow_bijector)
            losses -= tf.reduce_mean(flow.log_prob(z_mus))
            z_sample = tf.concat((states, prior.sample()), -1)
            for bijector in reversed(flow.bijector.bijectors):
                z_sample = bijector.forward(z_sample)
        elif variational == 'MAF':
            prior = tfd.MultivariateNormalDiag(
                loc=z_mus, scale_diag=tf.exp(z_log_sigmas))
            n_made_code = max(n_code + 1, n_made_code_factor * n_code)
            bijectors = []
            for i in range(n_bijectors):
                bijector = tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.
                    masked_autoregressive_default_template(
                        hidden_layers=[n_made_code] * n_made_layers))
                bijectors.append(
                    tfb.Permute(permutation=list(range(n_code - 1, -1, -1))))
            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
            flow = tfd.TransformedDistribution(
                distribution=prior, bijector=flow_bijector)
            losses -= tf.reduce_mean(flow.log_prob(z_mus))
            z_sample = tf.concat((states, prior.sample()), -1)
            for bijector in reversed(flow.bijector.bijectors):
                z_sample = bijector.forward(z_sample)
        elif variational == 'VQ':
            raise NotImplementedError('Nope.')
        else:
            raise NotImplementedError('Nope.')
        z_dec = tfl.dense(z_sample, state_size)
        if isinstance(initial_state[0], rnn.LSTMStateTuple):
            z_dec_rsz = tf.reshape(z_dec, [2, batch_size, n_neurons, n_layers])
            z = tuple(
                rnn.LSTMStateTuple(*[
                    tf.reshape(c, [batch_size, n_neurons]) for c in tf.split(
                        tf.reshape(el, [2, batch_size, n_neurons]), 2, axis=0)
                ]) for el in tf.split(z_dec_rsz, n_layers, axis=-1))
        else:
            z_dec_rsz = tf.reshape(z_dec, [batch_size, n_neurons, n_layers])
            z = tuple(
                tf.reshape(el, [batch_size, n_neurons])
                for el in tf.split(z_dec_rsz, n_layers, axis=-1))
    else:
        z = initial_state

    decoding_inputs_embed = tf.nn.dropout(
        input_layer(decoding_inputs), dec_in_keep_prob)

    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=decoding_inputs_embed,
        sequence_length=decoding_lengths,
        time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells, helper=helper, initial_state=z, output_layer=output_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_sequence_size)

    if model == 'mdn':
        helper = MDNRegressionHelper(
            batch_size=batch_size,
            max_sequence_size=synth_length,
            n_ch=n_ch,
            n_mixtures=n_mixtures,
            embed=input_layer)
    elif model == 'dml':
        helper = DMLRegressionHelper(
            batch_size=batch_size,
            max_sequence_size=synth_length,
            n_ch=n_ch,
            n_mixtures=n_mixtures,
            embed=input_layer)
    else:
        raise NotImplementedError('Not implemented.')

    scope.reuse_variables()
    infer_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells, helper=helper, initial_state=z, output_layer=output_layer)
    infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        infer_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=synth_length)
    # infer_logits = tf.identity(infer_outputs.sample_id, name='infer_logits')
    return outputs, infer_outputs, losses


def create_model(batch_size=1,
                 sequence_length=16000,
                 n_ch=1,
                 n_neurons=64,
                 n_layers=5,
                 input_embed_size=None,
                 target_embed_size=None,
                 n_mixtures=10,
                 model='mdn',
                 use_vq=False,
                 encode_stft=False,
                 cell_type='intersection',
                 wrapper='highway',
                 variational='MAF',
                 n_code=32,
                 n_bijectors=10,
                 n_made_layers=3,
                 n_made_code_factor=2,
                 use_attention=True):
    # [batch_size, max_time, n_ch]
    source = tf.placeholder(
        tf.float32, shape=(batch_size, sequence_length, n_ch), name='source')
    target = tf.placeholder(
        tf.float32, shape=(batch_size, sequence_length, n_ch), name='target')
    synth_length = tf.placeholder(tf.int32, shape=(), name='synth_length')

    if encode_stft:
        p = np.reshape(
            np.linspace(0.0, sequence_length - 1, sequence_length),
            [sequence_length, 1])
        k = np.reshape(
            np.linspace(0.0, 2 * np.pi / n_fft * (n_fft // 2), n_fft // 2),
            [1, n_fft // 2])
        freqs = tf.constant(np.dot(p, k))
        inputs, net = compute_inputs(x, freqs, n_fft, n_frames, input_features,
                                     norm)
        raise NotImplementedError('guh')

    lengths = tf.multiply(
        tf.ones((batch_size,), tf.int32),
        sequence_length,
        name='source_lengths')

    # Dropout
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
    dec_in_keep_prob = tf.placeholder_with_default(
        1.0, shape=(), name='dec_in_keep_prob')
    dec_in_noise_amt = tf.placeholder_with_default(
        0.0, shape=(), name='dec_in_noise_amt')
    beta = tf.placeholder_with_default(1.0, shape=(), name='beta')

    with tf.variable_scope('target/slicing'):
        source_last = tf.slice(source, [0, sequence_length - 1, 0],
                               [batch_size, 1, n_ch])
        decoder_input = tf.slice(target, [0, 0, 0],
                                 [batch_size, sequence_length - 1, n_ch])
        decoder_input = tf.concat([source_last, decoder_input], axis=1)
        # TODO: logarithmic frequency scaling of noise
        decoder_input = decoder_input + tf.random_normal(
            shape=[batch_size, sequence_length, n_ch],
            stddev=0.65) * dec_in_noise_amt
        decoder_output = tf.slice(target, [0, 0, 0],
                                  [batch_size, sequence_length, n_ch])

    input_layer = tfl.Dense(
        n_neurons,
        kernel_initializer=tf.initializers.variance_scaling(
            distribution='uniform'),
        name='input_projection')

    if input_embed_size:
        with tf.variable_scope('source/embedding'):
            source_embed, source_embed_matrix = _create_embedding(
                x=source, embed_size=input_embed_size)
    else:
        source_embed = source

    # Build the encoder
    with tf.variable_scope('encoder'):
        encoder_outputs, encoder_state = _create_encoder(
            source=source_embed,
            lengths=lengths,
            batch_size=batch_size,
            n_neurons=n_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob,
            cell_type=cell_type,
            wrapper=wrapper,
            input_layer=input_layer)

    # Build the decoder
    with tf.variable_scope('decoder') as scope:
        outputs, infer_outputs, var_loss = _create_decoder(
            n_neurons=n_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob,
            dec_in_keep_prob=dec_in_keep_prob,
            batch_size=batch_size,
            encoder_outputs=encoder_outputs,
            encoder_state=encoder_state,
            encoder_lengths=lengths,
            decoding_inputs=decoder_input,
            decoding_lengths=lengths,
            n_ch=n_ch,
            scope=scope,
            synth_length=synth_length,
            max_sequence_size=sequence_length,
            n_mixtures=n_mixtures,
            model=model,
            cell_type=cell_type,
            wrapper=wrapper,
            variational=variational,
            n_code=n_code,
            n_bijectors=n_bijectors,
            n_made_layers=n_made_layers,
            n_made_code_factor=n_made_code_factor,
            use_attention=use_attention,
            input_layer=input_layer)

    if model == 'mdn':
        max_sequence_size = sequence_length
        with tf.variable_scope('mdn'):
            means = tf.reshape(
                tf.slice(outputs[0], [0, 0, 0],
                         [batch_size, max_sequence_size, n_ch * n_mixtures]),
                [batch_size, max_sequence_size, n_ch, n_mixtures])
            sigmas = tf.minimum(
                10000.0,
                tf.maximum(
                    1e-1,
                    tf.nn.softplus(
                        tf.reshape(
                            tf.slice(outputs[0], [0, 0, n_ch * n_mixtures], [
                                batch_size, max_sequence_size, n_ch * n_mixtures
                            ]),
                            [batch_size, max_sequence_size, n_ch, n_mixtures
                            ]))))
            weights = tf.maximum(
                0.0,
                tf.nn.softmax(
                    tf.reshape(
                        tf.slice(outputs[0],
                                 [0, 0, n_ch * n_mixtures + n_ch * n_mixtures],
                                 [batch_size, max_sequence_size, n_mixtures]),
                        [batch_size, max_sequence_size, n_mixtures])))
            components = []
            for gauss_i in range(n_mixtures):
                mean_i = means[:, :, :, gauss_i]
                sigma_i = sigmas[:, :, :, gauss_i]
                comp_i = tfd.MultivariateNormalDiag(
                    loc=mean_i, scale_diag=sigma_i)
                components.append(comp_i)
            gauss = tfd.Mixture(
                cat=tfd.Categorical(probs=weights), components=components)
            sample = gauss.sample()
            tf.summary.histogram('means', means)
            tf.summary.histogram('sigmas', sigmas)
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('output', decoder_output)

        with tf.name_scope('loss'):
            negloglike = -gauss.log_prob(decoder_output)
            weighted_reconstruction = tf.reduce_sum(
                tf.expand_dims(weights, 2) * means, 3)
            max_reconstruction = tf.reduce_sum(means * tf.expand_dims(
                tf.one_hot(tf.argmax(weights, -1), n_mixtures), 2), 3)
            mix_loss = tf.reduce_mean(tf.reduce_mean(negloglike, 1), 0)
            mse_loss = tf.losses.mean_squared_error(weighted_reconstruction,
                                                    decoder_output)
            max_loss = tf.losses.mean_squared_error(max_reconstruction,
                                                    decoder_output) * 0.02
            rms_loss = tf.losses.mean_squared_error(
                tf.reduce_mean(weighted_reconstruction, -1),
                tf.reduce_mean(decoder_output, -1)) * 0.05
            loss = mix_loss + beta * var_loss + max_loss + rms_loss
            tf.summary.scalar('negloglike', tf.reduce_mean(negloglike))
            tf.summary.scalar('mix_loss', tf.reduce_mean(mix_loss))
            tf.summary.scalar('max_loss', tf.reduce_mean(max_loss))
            tf.summary.scalar('rms_loss', tf.reduce_mean(rms_loss))
            tf.summary.scalar('loss', tf.reduce_mean(loss))
    elif model == 'dml':
        with tf.name_scope('loss'):
            pred = tf.reshape(
                outputs[0],
                [batch_size * sequence_length, n_ch, 3 * n_mixtures])
            actual = tf.reshape(decoder_output,
                                [batch_size * sequence_length, n_ch])
            loc, unconstrained_scale, logits = tf.split(
                pred, num_or_size_splits=3, axis=-1)
            loc = tf.minimum(tf.nn.softplus(loc), 2.0**16 - 1.0)
            scale = tf.minimum(14.0,
                               tf.maximum(1e-8,
                                          tf.nn.softplus(unconstrained_scale)))
            discretized_logistic_dist = tfd.QuantizedDistribution(
                distribution=tfd.TransformedDistribution(
                    distribution=tfd.Logistic(loc=loc, scale=scale),
                    bijector=tfb.AffineScalar(shift=-0.5)),
                low=0.0,
                high=2**16 - 1.0)
            mixture_dist = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=discretized_logistic_dist)
            mix_loss = negloglike = -tf.reduce_sum(
                mixture_dist.log_prob((actual + 1.0) / 2.0 * (2**16 - 1)))
            sample = mixture_dist.sample()
            sample = tf.maximum(-1.0,
                                tf.minimum(1.0,
                                           sample / (2**16 - 1.0) * 2.0 - 1.0))
            max_loss = mse_loss = tf.losses.mean_squared_error(sample, actual)
            rms_loss = tf.losses.mean_squared_error(
                tf.reduce_mean(sample, -1), tf.reduce_mean(actual, -1))
            loss = beta * var_loss + negloglike
            weighted_reconstruction = sample
            # tf.summary.scalar('rms_loss', tf.reduce_mean(rms_loss))
            tf.summary.histogram('loc', loc)
            tf.summary.histogram('unconstrained_scale', unconstrained_scale)
            tf.summary.histogram('scale', scale)
            tf.summary.histogram('logits', logits)
            tf.summary.histogram('target', actual)
            tf.summary.histogram('sample', sample)
            # tf.summary.scalar('negloglike', negloglike)
            # tf.summary.scalar('max_loss', tf.reduce_mean(max_loss))
            tf.summary.scalar('mix_loss', mix_loss)
            tf.summary.scalar('loss', tf.reduce_mean(loss))
    else:
        with tf.name_scope('loss'):
            mix_loss = tf.reduce_mean(tf.reduce_sum([[0.0]], 1))
            mse_loss = tf.losses.mean_squared_error(outputs[0], decoder_output)
            max_loss = mse_loss
            loss = mse_loss + beta * var_loss
            tf.summary.scalar('mse_loss', tf.reduce_mean(mse_loss))
            tf.summary.scalar('loss', tf.reduce_mean(loss))
            sample = outputs[0]
            weighted_reconstruction = outputs[0]

    return {
        'source': source,
        'target': target,
        'keep_prob': keep_prob,
        'dec_in_keep_prob': dec_in_keep_prob,
        'dec_in_noise_amt': dec_in_noise_amt,
        'encoding': encoder_state,
        'decoding': infer_outputs,
        'sample': sample,
        'weighted': weighted_reconstruction,
        'loss': loss,
        'mix_loss': mix_loss,
        'mse_loss': mse_loss,
        'max_loss': max_loss,
        'var_loss': var_loss,
        'synth_length': synth_length,
        'beta': beta
    }


def batch_generator(data, sequence_length, target_offset, batch_size):
    n_batches = max(1, len(data) // sequence_length)
    for batch_i in range(n_batches):
        source, target = [], []
        while len(source) < batch_size:
            idx = np.random.randint(0,
                                    len(data) - sequence_length - target_offset)
            source.append(data[idx:idx + sequence_length])
            target.append(
                data[idx + target_offset:idx + target_offset + sequence_length])
        yield np.array(source), np.array(target)


def train(data,
          n_epochs=1000,
          batch_size=100,
          sequence_length=240,
          target_offset=240,
          ckpt_path='./',
          model_name='seq2seq.ckpt',
          restore_name=None,
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = create_model(
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_ch=data.shape[-1],
        **kwargs)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.001, global_step, 100, 0.96, staircase=True)
    noise_amt = tf.train.exponential_decay(0.1, global_step, 100, 0.96)
    tf.summary.scalar('noise', noise_amt)
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter(ckpt_path, sess.graph)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(ckpt_path):
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    step = 0
    for epoch_i in range(n_epochs):
        total, total_mse, total_mix, total_max, total_var = 0.0, 0.0, 0.0, 0.0, 0.0
        for it_i, (source, target) in enumerate(
                batch_generator(
                    data,
                    sequence_length=sequence_length,
                    target_offset=target_offset,
                    batch_size=batch_size)):
            na, lr = sess.run([noise_amt, learning_rate])
            summaries, step, mse_loss, mix_loss, max_loss, var_loss, _ = sess.run(
                [
                    merge, global_step, net['mse_loss'], net['mix_loss'],
                    net['max_loss'], net['var_loss'], opt
                ],
                feed_dict={
                    global_step: step + 1,
                    net['keep_prob']: 0.85,
                    net['dec_in_noise_amt']: na,
                    net['source']: source,
                    net['target']: target
                })
            total += mse_loss + mix_loss + max_loss + var_loss
            total_mix += mix_loss
            total_mse += mse_loss
            total_max += max_loss
            total_var += var_loss
            writer.add_summary(summaries, step)
            print('{}: mdn: {}, mse: {}, '
                  'max: {}, var: {} total: {}'.format(step.tolist(), mix_loss,
                                                      mse_loss, max_loss,
                                                      var_loss, total))
        print('\n-- epoch {}: mdn: {}, mse: {}, '
              'max: {}, total: {} --\n'.format(
                  epoch_i, total_mix / (it_i + 1), total_mse / (it_i + 1),
                  total_max / (it_i + 1), total / (it_i + 1)))
        saver.save(
            sess,
            os.path.join(ckpt_path, model_name),
            global_step=step.tolist())

    sess.close()


def _decode(example_proto):
    features = {'audio': tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return tf.decode_raw(parsed_features['audio'], tf.float32)


def preprocess(audio,
               sequence_length,
               target_offset,
               max_sequence_length=80000,
               scale='audio',
               frame_length=2048,
               hop_length=512,
               fft_length=2048):
    audio = audio / tf.reduce_max(tf.abs(audio))
    length = tf.shape(audio)[0]
    if scale == 'audio':
        data = tf.expand_dims(audio, -1)
        n_frames = max_sequence_length
    else:
        stft = tf.contrib.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=hop_length,
            fft_length=fft_length,
            window_fn=partial(tf.contrib.signal.hann_window, periodic=True),
            pad_end=True)
        if scale == 'energy':
            data = tf.cast(tf.abs(stft), tf.float32)
        elif scale == 'power':
            data = tf.cast(
                tf.real(stft) * tf.real(stft) + tf.imag(stft) * tf.imag(stft),
                tf.float32)
        elif scale == 'log-power':
            data = 20.0 * tf.log(1e-10 + tf.cast(
                tf.real(stft) * tf.real(stft) + tf.imag(stft) * tf.imag(stft),
                tf.float32))
        elif scale == 'normalized-log-power':
            data = 20.0 * tf.log(1e-10 + tf.cast(
                tf.real(stft) * tf.real(stft) + tf.imag(stft) * tf.imag(stft),
                tf.float32))
            data = tf.minimum(1.0, tf.maximum(0.0, (data + 450.0) / 700.0))
        else:
            raise ValueError('scale: {} not supported'.format(scale))
        n_frames = max_sequence_length // hop_length - (
            frame_length // hop_length)
    start = tf.random_uniform(
        (),
        maxval=n_frames - sequence_length - target_offset - 1,
        dtype=tf.int32)
    return data[start:start + sequence_length], data[
        start + target_offset:start + target_offset + sequence_length]


def train_tfrecords(files,
                    n_epochs=1000,
                    batch_size=100,
                    sequence_length=48000,
                    noise_factor=0.98,
                    target_offset=64,
                    frame_length=2048,
                    fft_length=2048,
                    hop_length=512,
                    n_ch=1,
                    scale='audio',
                    sr=16000,
                    ckpt_path='./',
                    model_name='seq2seq.ckpt',
                    restore_name=None,
                    **kwargs):
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_decode)
    dataset = dataset.filter(lambda x: tf.equal(tf.shape(x)[0], sr * 5))
    dataset = dataset.map(lambda x: preprocess(
        x, sequence_length=sequence_length, target_offset=target_offset,
        frame_length=frame_length, fft_length=fft_length, hop_length=hop_length,
        scale=scale))
    dataset = dataset.filter(
        lambda x, y: tf.equal(tf.shape(x)[0], sequence_length))
    dataset = dataset.shuffle(1000 + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat(n_epochs)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = create_model(
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_ch=n_ch,
        **kwargs)
    global_step = tf.Variable(
        1, name='global_step', trainable=False, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(
        0.002, global_step, 200, 0.998, staircase=True)
    # TODO: faster annealing
    noise_amt = tf.train.exponential_decay(0.1, global_step, 1000, noise_factor)
    tf.summary.scalar('noise', noise_amt)
    # TODO: higher weight norm
    weight_decay_amt = tf.constant(0.001)
    with tf.variable_scope('weights_norm') as scope:
        weights_norm = tf.reduce_sum(
            weight_decay_amt * tf.stack([
                tf.nn.l2_loss(i)
                for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if 'bias' not in i.name and 'Bias' not in i.name
            ]),
            name='weights_norm')
    tf.summary.scalar('weights_norm', weights_norm)
    # beta = tf.train.exponential_decay(0.9, global_step, 100, 0.98)
    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter(ckpt_path, sess.graph)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss'] + weights_norm, global_step=global_step)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint(ckpt_path):
        print('Restoring: ', ckpt_path)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

    n_batches = len(files)
    step = 0
    for epoch_i in range(n_epochs):
        total, total_var, total_mix, total_max = 0.0, 0.0, 0.0, 0.0
        for it_i in range(n_batches):
            source, target = sess.run(next_batch)
            if len(source) != batch_size or len(target) != batch_size:
                continue
            na, lr = sess.run([noise_amt, learning_rate])
            # lr = sess.run(learning_rate)
            summaries, step, var_loss, mix_loss, max_loss, _ = sess.run(
                [
                    merge, global_step, net['var_loss'], net['mix_loss'],
                    net['max_loss'], opt
                ],
                feed_dict={
                    net['keep_prob']: 0.85,
                    net['dec_in_noise_amt']: na,
                    net['source']: source,
                    net['target']: target
                })
            total += var_loss + mix_loss + max_loss
            total_mix += mix_loss
            total_var += var_loss
            total_max += max_loss
            print(
                '{}: mix: {}, var: {}, max: {}, total: {}'.format(
                    step.tolist(), mix_loss, var_loss, max_loss,
                    mix_loss + var_loss),
                end='\r')
            if step.tolist() % 10 == 0:
                writer.add_summary(summaries, step.tolist())

            if step.tolist() % 200 == 0:
                print('\nit: {}, lr: {}'.format(step.tolist(), lr))
                print(
                    '\n-- epoch {}: mix: {}, var: {}, max: {}, total: {} --\n'.
                    format(epoch_i, total_mix / (it_i + 1),
                           total_var / (it_i + 1), total_max / (it_i + 1),
                           total / (it_i + 1)))
                saver.save(
                    sess,
                    os.path.join(ckpt_path, model_name),
                    global_step=global_step)

    sess.close()


def infer(source, target, synth_length, ckpt_path='./', **kwargs):
    if source.ndim == 2:
        source = source[np.newaxis]
        target = target[np.newaxis]
    batch_size = source.shape[0]
    sequence_length = source.shape[1]
    n_ch = source.shape[2]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        net = create_model(
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_ch=n_ch,
            **kwargs)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        saver.restore(sess, ckpt)
        recon, enc = sess.run(
            [net['decoding'], net['encoding']],
            feed_dict={
                net['source']: source,
                net['synth_length']: synth_length
            })
        src, tgt = source, target
        res = recon[1]
        fig, axs = plt.subplots(1, 3, sharey=True)
        axs[0].plot(src.reshape(-1, src.shape[-1]))
        axs[0].set_title('Source')
        axs[1].plot(tgt.reshape(-1, tgt.shape[-1]))
        axs[1].set_title('Target (Original)')
        axs[2].plot(res.reshape(-1, res.shape[-1]))
        axs[2].set_title('Target (Synthesis Sampling)')
        return {
            'source': src,
            'target': tgt,
            'encoding': enc,
            'prediction': res
        }
