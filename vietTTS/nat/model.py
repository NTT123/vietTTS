import math
from operator import pos

import haiku as hk
import jax
import jax.numpy as jnp
from jax.numpy import ndarray

from .config import FLAGS, AcousticInput
from .data_loader import frame_idx_encode


class BiLSTM(hk.Module):
  def __init__(self, lstm_dim, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.lstm_fwd = hk.LSTM(lstm_dim)
    self.lstm_bwd = hk.ResetCore(hk.LSTM(lstm_dim))

  def __call__(self, x, lengths):
    B, L, D = x.shape
    mask = jnp.arange(0, L)[None, :] >= (lengths[:, None] - 1)
    h0c0_fwd = self.lstm_fwd.initial_state(B)
    new_hx_fwd, new_hxcx_fwd = hk.dynamic_unroll(self.lstm_fwd, x, h0c0_fwd, time_major=False)
    x_bwd, mask_bwd = jax.tree_map(lambda x: jnp.flip(x, axis=1), (x, mask))
    h0c0_bwd = self.lstm_bwd.initial_state(B)
    new_hx_bwd, new_hxcx_bwd = hk.dynamic_unroll(self.lstm_bwd, (x_bwd, mask_bwd), h0c0_bwd, time_major=False)
    x = jnp.concatenate((new_hx_fwd, jnp.flip(new_hx_bwd, axis=1)), axis=-1)
    return x


def positional_encoder(x, position, dim):
  position = position[:, :, None]
  div_term = jnp.exp(jnp.arange(0, dim, 2, dtype=jnp.float32) * (-math.log(10_000.0) / dim))
  div_term = div_term[None, None, :]
  enc1 = jnp.sin(position * div_term)
  enc2 = jnp.cos(position * div_term)
  return jnp.concatenate((x, enc1, enc2), axis=-1)


class TokenEncoder(hk.Module):
  """Encode phonemes/text to vector"""

  def __init__(self, vocab_size, lstm_dim, dropout_rate, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.embed = hk.Embed(vocab_size, lstm_dim)
    self.conv1 = hk.Conv1D(lstm_dim, 3, padding='SAME')
    self.conv2 = hk.Conv1D(lstm_dim, 3, padding='SAME')
    self.conv3 = hk.Conv1D(lstm_dim, 3, padding='SAME')
    self.bn1 = hk.BatchNorm(True, True, 0.999)
    self.bn2 = hk.BatchNorm(True, True, 0.999)
    self.bn3 = hk.BatchNorm(True, True, 0.999)
    self.bilstm = BiLSTM(lstm_dim, is_training)
    self.dropout_rate = dropout_rate

  def __call__(self, x, lengths):
    x = self.embed(x)
    x = jax.nn.relu(self.bn1(self.conv1(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = jax.nn.relu(self.bn2(self.conv2(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = jax.nn.relu(self.bn3(self.conv3(x), is_training=self.is_training))
    x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if self.is_training else x
    x = self.bilstm(x, lengths)
    return x


class ScalarPredictor(hk.Module):
  """Range/Duration Predictor."""

  def __init__(self, lstm_dim, use_lstm=True, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.use_lstm = use_lstm
    if use_lstm:
      self.bilstm1 = BiLSTM(lstm_dim, is_training)
      self.bilstm2 = BiLSTM(lstm_dim, is_training)
    self.projection = hk.Linear(1)

  def __call__(self, x, lengths):
    if self.use_lstm:
      x = self.bilstm1(x, lengths)
      x = self.bilstm2(x, lengths)
    x = jnp.squeeze(self.projection(x), axis=-1)
    x = jax.nn.softplus(x)
    return x


class NATNet(hk.Module):
  """Predict melspectrogram from phonemes"""

  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training
    self.encoder = TokenEncoder(FLAGS.vocab_size, FLAGS.acoustic_encoder_dim, 0.5, is_training)
    self.speaker_embed = hk.Embed(256, FLAGS.speaker_embed_dim,
                                  w_init=hk.initializers.TruncatedNormal(1.0 / FLAGS.speaker_embed_dim))
    self.decoder = hk.deep_rnn_with_skip_connections([
        hk.LSTM(FLAGS.acoustic_decoder_dim),
        hk.LSTM(FLAGS.acoustic_decoder_dim)
    ])
    self.projection = hk.Linear(FLAGS.mel_dim)

    # prenet
    self.prenet_fc1 = hk.Linear(256, with_bias=False)
    self.prenet_fc2 = hk.Linear(256, with_bias=False)
    # posnet
    self.postnet_convs = [hk.Conv1D(FLAGS.postnet_dim, 5) for _ in range(4)] + [hk.Conv1D(FLAGS.mel_dim, 5)]
    self.postnet_bns = [hk.BatchNorm(True, True, 0.999) for _ in range(4)] + [None]

    # upsample
    self.duration_predictor = ScalarPredictor(FLAGS.duration_lstm_dim, False, is_training)
    self.range_predictor = ScalarPredictor(FLAGS.range_lstm_dim, True, is_training)

  def prenet(self, x, dropout=0.5):
    x = jax.nn.relu(self.prenet_fc1(x))
    x = hk.dropout(hk.next_rng_key(), dropout, x)
    x = jax.nn.relu(self.prenet_fc2(x))
    x = hk.dropout(hk.next_rng_key(), dropout, x)
    return x

  def upsample(self, x, durations, ranges, L):
    ruler = jnp.arange(0, L)[None, :]  # B, L
    durations = durations * FLAGS.sample_rate / FLAGS.hop_length
    end_pos = jnp.cumsum(durations, axis=1)
    mid_pos = end_pos - durations/2  # B, T

    d2 = jnp.square((mid_pos[:, None, :] - ruler[:, :, None]) / ranges[:, None, :]) / 10.0
    w = jax.nn.softmax(-d2, axis=-1)
    hk.set_state('attn', w)
    x = jnp.einsum('BLT,BTD->BLD', w, x)
    return x

  def postnet(self, mel: ndarray) -> ndarray:
    x = mel
    for conv, bn in zip(self.postnet_convs, self.postnet_bns):
      x = conv(x)
      if bn is not None:
        x = bn(x, is_training=self.is_training)
        x = jnp.tanh(x)
      x = hk.dropout(hk.next_rng_key(), 0.5, x) if self.is_training else x
    return x

  def inference(self, tokens, silence_duration, speaker):
    B, L = tokens.shape
    lengths = jnp.array([L], dtype=jnp.int32)
    x = self.encoder(tokens, lengths)
    speaker = jnp.array([speaker], dtype=jnp.int32)
    speaker = self.speaker_embed(speaker)[:, None, :]
    speaker = jnp.broadcast_to(speaker, (speaker.shape[0], x.shape[1], speaker.shape[-1]))
    x = jnp.concatenate((x, speaker), axis=-1)
    durations = self.duration_predictor(x, lengths)
    durations = jnp.where(
        tokens == FLAGS.sp_index,
        jnp.clip(durations, a_min=silence_duration, a_max=None),
        durations
    )
    durations = jnp.where(tokens == FLAGS.word_end_index, 0., durations)
    n_frames = jnp.sum(durations) * FLAGS.sample_rate / FLAGS.hop_length
    range_inputs = jnp.concatenate((x, durations[..., None]), axis=-1)
    ranges = self.range_predictor(range_inputs, lengths)
    x = self.upsample(x, durations, ranges, n_frames)
    B, L, D = x.shape
    durations = jax.device_get(durations[0]).tolist()
    frame_idx_fwd = frame_idx_encode(durations, forward=True)
    frame_idx_fwd = jnp.array(frame_idx_fwd)[None, :]
    frame_idx_bwd = frame_idx_encode(durations, forward=False)
    frame_idx_bwd = jnp.array(frame_idx_bwd)[None, :]
    x = positional_encoder(x, frame_idx_fwd[:, :L], 16)
    x = positional_encoder(x, frame_idx_bwd[:, :L], 16)

    def loop_fn(inputs, state):
      cond = inputs
      prev_mel, hxcx = state
      prev_mel = self.prenet(prev_mel)
      x = jnp.concatenate((cond, prev_mel), axis=-1)
      x, new_hxcx = self.decoder(x, hxcx)
      x = jnp.concatenate((x, cond), axis=-1)
      x = self.projection(x)
      return x, (x, new_hxcx)

    state = (
        jnp.zeros((B, FLAGS.mel_dim), dtype=jnp.float32),
        self.decoder.initial_state(B)
    )
    x, _ = hk.dynamic_unroll(loop_fn, x, state, time_major=False)
    residual = self.postnet(x)
    return x + residual

  def __call__(self, inputs: AcousticInput):
    x = self.encoder(inputs.phonemes, inputs.lengths)
    speaker = self.speaker_embed(inputs.speakers)[:, None, :]
    speaker = jnp.broadcast_to(speaker, (speaker.shape[0], x.shape[1], speaker.shape[-1]))
    x = jnp.concatenate((x, speaker), axis=-1)
    duration_hat = self.duration_predictor(x, inputs.lengths)
    range_inputs = jnp.concatenate((x, inputs.durations[..., None]), axis=-1)
    ranges = self.range_predictor(range_inputs, inputs.lengths)
    x = self.upsample(x, inputs.durations, ranges, inputs.mels.shape[1])
    x = positional_encoder(x, inputs.frame_idx_fwd, 16)
    x = positional_encoder(x, inputs.frame_idx_bwd, 16)
    cond = x

    mels = self.prenet(inputs.mels)
    x = jnp.concatenate((x, mels), axis=-1)
    B, L, D = x.shape
    hx = self.decoder.initial_state(B)

    def zoneout_decoder(inputs, prev_state):
      x, mask = inputs
      x, state = self.decoder(x, prev_state)
      state = jax.tree_multimap(lambda m, s1, s2: s1*m + s2*(1-m), mask, prev_state, state)
      return x, state

    mask = jax.tree_map(lambda x: jax.random.bernoulli(hk.next_rng_key(), 0.1, (B, L, x.shape[-1])), hx)
    x, _ = hk.dynamic_unroll(zoneout_decoder, (x, mask), hx, time_major=False)
    x = jnp.concatenate((x, cond), axis=-1)
    x = self.projection(x)
    residual = self.postnet(x)
    return x, x + residual, duration_hat
