import pickle

import jax
import jax.numpy as jnp
from numpy import ndarray

from .config import *
from .model import *


def encode_text(text):
  out = []
  for c in text:
    if c in FLAGS._alphabet:
      out.append(FLAGS._alphabet.index(c))
  return out


@hk.transform_with_state
def generate_mel(text, reduce_factor=1):
  net = Tacotron(is_training=False)
  text_len = jnp.array([text.shape[1]])
  text = net.encode_text(text, text_len)
  N, L, D = text.shape

  def loop_fn(inputs, prev_states):
    rng = inputs
    mel, attn_lstm_hxcx, decoder_lstm_hxcx, att_ctx, attn_loc = prev_states
    mel = net.prenet(mel, 0.0)
    # mel = hk.dropout(rng, 0.5, mel)
    x = jnp.concatenate((mel, att_ctx), axis=-1)
    x, new_attn_lstm_hxcx = net.attn_lstm(x, attn_lstm_hxcx)
    attn_hx = x
    # attention -> new context
    attn_params = net.attn_V(jnp.tanh(net.attn_W(x)))
    delta, scale = jnp.split(attn_params, 2, axis=-1)
    delta = jax.nn.softplus(delta) / 2 * reduce_factor
    # google GMMA paper suggests std = 10
    scale = jax.nn.softplus(scale + 5)
    new_loc = attn_loc + delta
    j = jnp.arange(0, L)[None, :]
    up = (j - new_loc + 0.5) / scale
    down = (j - new_loc - 0.5) / scale
    weight = jax.nn.sigmoid(up) - jax.nn.sigmoid(down)  # N L
    new_context = jnp.sum(weight[..., None] * text, axis=1)  # N D

    x = jnp.concatenate((attn_hx, new_context), axis=-1)
    output, new_decoder_lstm_hxcx = net.decoder_lstm(x, decoder_lstm_hxcx)

    dec = net.dec_proj(output)
    dec = rearrange(dec, 'N (R D) -> N R D', R=FLAGS.max_reduce_factor, D=1 + FLAGS.mel_dim)
    dec = dec[:, :reduce_factor, :]
    stop_token, predicted_mel = jnp.split(dec, [1, ], axis=-1)
    mel = predicted_mel
    return (mel, stop_token), (mel[:, -1, :], new_attn_lstm_hxcx, new_decoder_lstm_hxcx, new_context, new_loc)

  initial_state = (
      jnp.zeros((N, FLAGS.mel_dim)),
      net.attn_lstm.initial_state(1),
      net.decoder_lstm.initial_state(1),
      jnp.zeros((N, D)),
      jnp.zeros((N, 1))
  )

  rng = jax.random.PRNGKey(42)
  rngs = jnp.stack(jax.random.split(rng, L*4), axis=1)
  (mel, stop), _ = hk.dynamic_unroll(loop_fn, rngs, initial_state, time_major=False)
  stop = jax.nn.sigmoid(stop)
  l = jnp.sum(stop < 0.5).item()
  mel = rearrange(mel, 'N L R D -> N (L R) D')
  mel = mel[:, :l, :]
  mel = mel + net.postnet(mel)
  # mel = mel + net.postnet(mel)
  return mel, jax.nn.sigmoid(stop)


def text2mel(text: str) -> ndarray:
  print('Loading latest tacotron checkpoint at ', FLAGS.ckpt_dir / 'latest_state.pickle')
  with open(FLAGS.ckpt_dir / 'latest_state.pickle', 'rb') as f:
    dic = pickle.load(f)
    start_step = dic['step'] + 1
    params = dic['params']
    aux = dic['aux']
    optim_state = dic['optim_state']

  rng = jax.random.PRNGKey(42)
  mel, stop = generate_mel.apply(
      params,
      aux,
      rng,
      jnp.array(encode_text(text))[None, :]
  )[0]
  return mel
