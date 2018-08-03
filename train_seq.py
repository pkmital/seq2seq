"""

Copyright Parag K. Mital 2018. All rights reserved.
"""
import os
import lws
import scipy
import glob
import librosa
from functools import partial
import seq2seq
import tensorflow as tf
import numpy as np
import click


@click.command()
@click.option('--config', default=1, help='Which config to load')
@click.option('--mode', default='infer', help='"train" or "infer"')
def run(config, mode):
    if config == 0:
        v = 'v4'
        batch_size = 128
        sr = 16000
        sequence_length = 64
        target_offset = 64
        n_layers = 12
        n_neurons = 256
        n_mixtures = 4
        n_ch = 513
        fft_length = 1024
        frame_length = 1024
        hop_length = 256
        model = 'mdn'
        cell_type = 'intersection'
        wrapper = 'highway'
        variational = 'MAF'
        noise_factor = 0.8
        n_code = 1024
        scale = 'power'
        n_bijectors = 6
        n_made_code_factor = 3
        n_made_layers = 3
        use_attention = False
        n_files = 1
    elif config == 1:
        v = 'v4'
        batch_size = 128
        sr = 16000
        sequence_length = 64
        target_offset = 64
        n_layers = 12
        n_neurons = 256
        n_mixtures = 8
        n_ch = 513
        fft_length = 1024
        frame_length = 1024
        hop_length = 256
        model = 'mdn'
        cell_type = 'intersection'
        wrapper = 'highway'
        variational = 'MAF'
        noise_factor = 0.8
        n_code = 1024
        scale = 'power'
        n_bijectors = 6
        n_made_code_factor = 3
        n_made_layers = 3
        use_attention = False
        n_files = 1
    elif config == 2:
        v = 'v4'
        batch_size = 32
        sr = 16000
        sequence_length = 128
        target_offset = 128
        n_layers = 12
        n_neurons = 256
        n_mixtures = 4
        n_ch = 1025
        fft_length = 2048
        frame_length = 2048
        hop_length = 256
        model = 'mdn'
        cell_type = 'intersection'
        wrapper = 'highway'
        variational = 'MAF'
        noise_factor = 0.8
        n_code = 1024
        scale = 'power'
        n_bijectors = 6
        n_made_code_factor = 3
        n_made_layers = 3
        use_attention = False
        n_files = 1
    elif config == 3:
        v = 'v4'
        batch_size = 128
        sr = 16000
        sequence_length = 64
        target_offset = 64
        n_layers = 12
        n_neurons = 256
        n_mixtures = 5
        n_ch = 513
        fft_length = 1024
        frame_length = 1024
        hop_length = 256
        model = 'mdn'
        cell_type = 'layer-norm-lstm'
        wrapper = 'highway'
        variational = None
        noise_factor = 0.8
        n_code = 8192
        scale = 'power'
        n_bijectors = 6
        n_made_code_factor = 3
        n_made_layers = 3
        use_attention = False
        n_files = 1
    else:
        raise NotImplementedError('Nope.')

    if v == 'v4':
        logdir = os.path.expanduser(
            './logs_iaf/phl_b={},nf={},n_ch={},sr={},seqlen={},ng={},'
            'nl={},nn={},ct={},w={},v={},nc={},att={},nbj={},nmcf={},'
            'nml={},m={},fft={},fr={},hop={},n={},scale={}'.
            format(batch_size, n_files, n_ch, sr, sequence_length, n_mixtures,
                   n_layers, n_neurons, cell_type, wrapper, variational, n_code,
                   use_attention, n_bijectors, n_made_code_factor,
                   n_made_layers, model, fft_length, frame_length, hop_length,
                   noise_factor, scale))
    elif v == 'v3':
        logdir = os.path.expanduser(
            './logs_iaf/phl_b={},nf={},sr={},seqlen={},ng={},nl={},nn={},'
            'ct={},w={},v={},nc={},att={},nbj={},nmcf={},nml={},m={},'
            'fft={},fr={},hop={},n={},scale={}'.
            format(batch_size, n_files, sr, sequence_length, n_mixtures,
                   n_layers, n_neurons, cell_type, wrapper, variational, n_code,
                   use_attention, n_bijectors, n_made_code_factor,
                   n_made_layers, model, fft_length, frame_length, hop_length,
                   noise_factor, scale))
    elif v == 'v2':
        logdir = os.path.expanduser(
            './logs_iaf/phl_b={},sr={},seqlen={},ng={},nl={},nn={},ct={},'
            'w={},v={},nc={},att={},nbj={},nmcf={},nml={},m={},fft={},fr={},'
            'hop={},n={}'.
            format(batch_size, sr, sequence_length, n_mixtures, n_layers,
                   n_neurons, cell_type, wrapper, variational, n_code,
                   use_attention, n_bijectors, n_made_code_factor,
                   n_made_layers, model, fft_length, frame_length, hop_length,
                   noise_factor))
    elif v == 'v1':
        logdir = os.path.expanduser(
            './logs_iaf/phl_b={},sr={},seqlen={},ng={},nl={},nn={},ct={},w={},'
            'v={},nc={},att={},nbj={},nmcf={},nml={},m={}'.
            format(batch_size, sr, sequence_length, n_mixtures, n_layers,
                   n_neurons, cell_type, wrapper, variational, n_code,
                   use_attention, n_bijectors, n_made_code_factor,
                   n_made_layers, model))
    else:
        raise ValueError('Unsupported version: {}'.format(v))
    print(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    files = [
        f for f in glob.glob('records/*.tfrecord')
        if os.stat(f).st_size > 0
    ]

    if mode == 'train':
        seq2seq.train_tfrecords(
            files[:n_files],
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_offset=target_offset,
            ckpt_path=logdir,
            scale=scale,
            model=model,
            noise_factor=noise_factor,
            n_ch=n_ch,
            frame_length=frame_length,
            fft_length=fft_length,
            hop_length=hop_length,
            n_mixtures=n_mixtures,
            n_layers=n_layers,
            n_neurons=n_neurons,
            cell_type=cell_type,
            wrapper=wrapper,
            variational=variational,
            n_code=n_code,
            use_attention=use_attention,
            n_bijectors=n_bijectors,
            n_made_code_factor=n_made_code_factor,
            n_made_layers=n_made_layers)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Graph().as_default(), tf.Session(config=config) as sess:
            dataset = tf.data.TFRecordDataset([files])
            dataset = dataset.shuffle(1000 + 3 * batch_size)
            dataset = dataset.map(seq2seq._decode)
            dataset = dataset.map(
                partial(
                    seq2seq.preprocess,
                    fft_length=fft_length,
                    hop_length=hop_length,
                    frame_length=frame_length,
                    sequence_length=sequence_length,
                    scale=scale,
                    target_offset=target_offset))
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()
            source, target = sess.run(next_batch)

        res = seq2seq.infer(
            source,
            target,
            ckpt_path=logdir,
            model=model,
            n_mixtures=n_mixtures,
            n_layers=n_layers,
            n_neurons=n_neurons,
            cell_type=cell_type,
            wrapper=wrapper,
            variational=variational,
            synth_length=sequence_length,
            n_code=n_code,
            use_attention=use_attention,
            n_bijectors=n_bijectors,
            n_made_code_factor=n_made_code_factor,
            n_made_layers=n_made_layers)

        with tf.Graph().as_default(), tf.Session(config=config) as sess:
            stfts_op = tf.placeholder(dtype=tf.complex64, name='stfts')
            istft_op = tf.contrib.signal.inverse_stft(
                stfts_op,
                frame_length=frame_length,
                frame_step=hop_length,
                fft_length=frame_length,
                window_fn=partial(tf.contrib.signal.hann_window, periodic=True))
            phase_ret = lws.lws(frame_length, hop_length)
            for pred_i, (src, tgt, pred) in enumerate(
                    zip(source, target, res['prediction'])):

                if scale == 'power':
                    src = np.sqrt(np.maximum(src, 1e-12))
                    tgt = np.sqrt(np.maximum(tgt, 1e-12))
                    pred = np.sqrt(np.maximum(pred, 1e-12))
                elif scale == 'log-power':
                    src = np.sqrt(np.maximum(1e-12, np.exp(src / 20.0)))
                    tgt = np.sqrt(np.maximum(1e-12, np.exp(tgt / 20.0)))
                    pred = np.sqrt(np.maximum(1e-12, np.exp(pred / 20.0)))
                elif scale == 'normalized-log-power':
                    src = np.sqrt(np.maximum(1e-12, np.exp(
                        ((src * 700.0) - 450.0) / 20.0)))
                    tgt = np.sqrt(np.maximum(1e-12, np.exp(
                        ((tgt * 700.0) - 450.0) / 20.0)))
                    pred = np.sqrt(np.maximum(1e-12, np.exp(
                        ((pred * 700.0) - 450.0) / 20.0)))

                if scale != 'audio':
                    stfts = phase_ret.run_lws(src.astype(np.float64)).astype(
                        np.complex64)
                    src = sess.run(istft_op, feed_dict={stfts_op: stfts})
                    stfts = phase_ret.run_lws(tgt.astype(np.float64)).astype(
                        np.complex64)
                    tgt = sess.run(istft_op, feed_dict={stfts_op: stfts})
                    stfts = phase_ret.run_lws(pred.astype(np.float64)).astype(
                        np.complex64)
                    pred = sess.run(istft_op, feed_dict={stfts_op: stfts})

                scipy.io.wavfile.write(
                    os.path.join(logdir, 'phl-src-{}.wav'.format(pred_i)), sr,
                    src.astype(np.float32))
                scipy.io.wavfile.write(
                    os.path.join(logdir, 'phl-tgt-{}.wav'.format(pred_i)), sr,
                    tgt.astype(np.float32))
                scipy.io.wavfile.write(
                    os.path.join(logdir, 'phl-synth-{}.wav'.format(pred_i)), sr,
                    pred.astype(np.float32))


if __name__ == '__main__':
    run()
